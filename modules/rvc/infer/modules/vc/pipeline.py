import logging
import os
import traceback
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal

from handlers.config import model_path
from handlers.noise_removal import restore_silence
from handlers.stereo import stereo_to_mono_ms, resample_side, mono_to_stereo_ms
from modules.rvc.infer.lib.audio import load_audio_advanced
from modules.rvc.infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, \
    SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from modules.rvc.infer.modules.vc.utils import check_faiss_index_file, extract_index_from_zip, cache_harvest_f0, \
    change_rms, debug_clone_audio, get_index_path_from_model, load_hubert

logger = logging.getLogger(__name__)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        # Minimal chunking tweaks: add +0.2s to x_pad, +2s to x_query
        self.x_pad = config.x_pad + 0.2
        self.x_query = config.x_query + 2
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half

        self.sr: int = 16000
        self.window: int = 160
        self.t_pad: int = int(self.sr * self.x_pad)
        self.t_pad_tgt: int = int(tgt_sr * self.x_pad)
        self.t_pad2: int = self.t_pad * 2
        self.t_query: int = int(self.sr * self.x_query)
        self.t_center: int = int(self.sr * self.x_center)
        self.t_max: int = int(self.sr * self.x_max)
        self.device = config.device

    def get_f0(
            self,
            input_audio_path,
            x,
            p_len,
            f0_up_key,
            f0_method,
            filter_radius,
            inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            batch_size = 512
            audio_t = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio_t,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        else:
            if not hasattr(self, "model_rmvpe"):
                from modules.rvc.infer.lib.rmvpe import RMVPE
                rvmpe_model_path = os.path.join(model_path, "rvc", "rmvpe.pt")
                logger.info(f"Loading RMVPE model from {rvmpe_model_path}")
                self.model_rmvpe = RMVPE(
                    model_path=rvmpe_model_path,
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            if "privateuseone" in str(self.device):
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")

        # pitch shift
        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window

        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (
                (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        )
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def voice_clone(
            self,
            model,
            net_g,
            sid,
            audio0,
            pitch,
            pitchf,
            times,
            index,
            big_npy,
            index_rate,
            version,
            protect,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # forced mono in original
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        if (
                index is not None
                and big_npy is not None
                and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                    + (1 - index_rate) * feats
            )

        # Log before and after upsampling
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(
                feats0.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)

        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            has_pitch = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if has_pitch else (feats, p_len, sid)
            audio1 = net_g.infer(*arg)[0][0, 0].data.cpu().float().numpy()

        # Output length and inferred SR
        inferred_sr = len(audio1) / len(audio0) * self.sr

        del has_pitch, arg, feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    # Yet another layer of functions!
    def pipeline(
            self,
            model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=None,
    ):
        # Helper function to print peak volume in dB
        if file_index is not None and os.path.exists(file_index) and index_rate != 0:
            try:
                if not check_faiss_index_file(file_index):
                    file_index = extract_index_from_zip(file_index, model_path)
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        og_audio = audio

        # 1) First modification: filtfilt
        audio = signal.filtfilt(bh, ah, audio)
        debug_clone_audio(audio, self.sr, "01_filtfilt")

        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i: i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query: t + self.t_query]
                        == audio_sum[t - self.t_query: t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()

        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        t2 = ttime()
        times[1] += t2 - t1

        # 2) Process all segments except the final chunk
        for idx, t in enumerate(opt_ts):
            t = t // self.window * self.window
            seg_base = audio_pad[s: t + self.t_pad2 + self.window]
            if if_f0 == 1:
                seg = self.voice_clone(
                    model,
                    net_g,
                    sid_tensor,
                    seg_base,
                    pitch[:, s // self.window: (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window: (t + self.t_pad2) // self.window],
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )
            else:
                seg = self.voice_clone(
                    model,
                    net_g,
                    sid_tensor,
                    seg_base,
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )
            seg = seg[self.t_pad_tgt: -self.t_pad_tgt]

            audio_opt.append(seg)
            s = t

        # 3) Final chunk
        if if_f0 == 1:
            final_seg = self.voice_clone(
                model,
                net_g,
                sid_tensor,
                audio_pad[s:],
                pitch[:, s // self.window:] if s else pitch,
                pitchf[:, s // self.window:] if s else pitchf,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )
        else:
            final_seg = self.voice_clone(
                model,
                net_g,
                sid_tensor,
                audio_pad[s:],
                None,
                None,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )

        final_seg = final_seg[self.t_pad_tgt: -self.t_pad_tgt]

        audio_opt.append(final_seg)
        audio_opt = np.concatenate(audio_opt)
        debug_clone_audio(audio_opt, self.sr, "02_after_vc_concat")

        # 4) Possibly change RMS
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
            debug_clone_audio(audio_opt, tgt_sr, "03_after_change_rms")

        # 5) Restore silence if needed
        if len(audio_opt) != len(og_audio):
            og_audio_resampled = librosa.resample(
                og_audio, orig_sr=self.sr, target_sr=(len(audio_opt) / len(og_audio)) * self.sr
            )
            debug_clone_audio(audio_opt, tgt_sr, "03b_restore_silence_resampled_og")
            logger.info(f"Resampled OG audio to match output length: {len(og_audio_resampled)}")
            audio_opt = restore_silence(og_audio_resampled, audio_opt)
        else:
            debug_clone_audio(audio_opt, tgt_sr, "03b_restore_silence_resampled_og")
            audio_opt = restore_silence(og_audio, audio_opt)
        debug_clone_audio(audio_opt, tgt_sr, "04_after_restore_silence")

        # 6) Resample to final sample rate if needed
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
            debug_clone_audio(audio_opt, resample_sr, "05_after_final_resample")

        # 7) Convert to int16
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            logger.info("Warning: audio_max > 1")
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        final_sr = resample_sr if tgt_sr != resample_sr and resample_sr >= 16000 else tgt_sr
        debug_clone_audio(audio_opt, final_sr, "06_after_int16_conversion")

        del pitch, pitchf, sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline: Pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None
        self.index = None

        self.config = config
        self.global_step = 0
        self.total_steps = 0

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                    self.hubert_model is not None
            ):
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", None)
                if not self.version:
                    # ckpt.config[14] will start with 24, 20 if v2, 16,16, if v1
                    sval = self.cpt["config"][14][0]
                    self.version = "v2" if sval == 24 else "v1"
                    logger.info(f"Version: {sval}, {self.version}")

                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                }
            )
        person = os.path.join(model_path, "trained", sid)
        # person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(person), "__type__": "update"}
        logger.info(f"Select index: {index}")
        self.index = index["value"]
        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect1
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"},
            "", ""
        )

    # This is called for each project by vc_multi
    def vc_single(
            self,
            model,
            sid,
            input_audio_path,
            f0_up_key,
            f0_file,
            f0_method,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            reverb_param_path=None,
            do_noise_removal=True,
            callback=None,
    ):
        """
        Processes a single audio file, resampling as needed for processing, and restoring the original sample rate.

        1) Loads the original audio in float[-1,1] at its native sample rate (og_sr).
        2) Resamples to 16k for RVC processing.
        3) Runs the pipeline (pitch extraction, chunk processing).
        4) Resamples the pipeline output back to og_sr.
        5) Optionally applies noise removal and reverb.
        6) Ensures correct RMS scaling before returning the final output.

        Returns:
            Tuple (success_message, (final_sample_rate, final_audio_float))
        """

        try:
            # ----------------------------------------------------------------
            # (A) Load original audio in float[-1,1] at its original sample rate (og_sr)
            # ----------------------------------------------------------------
            audio_float, og_sr = load_audio_advanced(
                file=input_audio_path,
                sr=None,  # Do NOT resample, keep original
                mono=False,  # Keep stereo if applicable
                return_sr=True
            )

            if callback is not None:
                callback(self.global_step / self.total_steps,
                         f"Loaded audio: shape={audio_float.shape}, original sample rate={og_sr}",
                         self.total_steps)
            logger.info(f"Loaded audio: shape={audio_float.shape}, original sample rate={og_sr}")

            # Normalize if necessary
            # audio_float /= max(1.0, np.abs(audio_float).max() / 0.95)

            # ----------------------------------------------------------------
            # (B) Handle Stereo -> Mono conversion (Mid-Side Encoding)
            # ----------------------------------------------------------------
            if audio_float.ndim == 2 and audio_float.shape[1] == 2:
                callback(self.global_step / self.total_steps, "Converting stereo to mono (Mid-Side Encoding)",
                         self.total_steps)
                logger.info("Converting stereo to mono (Mid-Side Encoding)")
                mono_og, side_og, orig_len = stereo_to_mono_ms(audio_float)
            else:
                logger.info("Audio is already mono")
                mono_og = audio_float if audio_float.ndim == 1 else audio_float.mean(axis=1)
                side_og = None
                orig_len = len(mono_og)

            # ----------------------------------------------------------------
            # (C) Resample to 16k for the RVC pipeline
            # ----------------------------------------------------------------
            sr_rvc = 16000

            if og_sr != sr_rvc:
                mono_for_pipeline = librosa.resample(mono_og, orig_sr=og_sr, target_sr=sr_rvc)
            else:
                mono_for_pipeline = mono_og

            # ----------------------------------------------------------------
            # (D) Run the pipeline at 16k
            # ----------------------------------------------------------------
            times = [0, 0, 0]
            file_index = self.index

            if self.pipeline is None:
                self.get_vc(model)

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            logger.info("Running pipeline...")
            if callback is not None:
                callback(self.global_step / self.total_steps, "Running pipeline...", self.total_steps)

            # Start the actual voice cloning...
            audio_opt_16k = self.pipeline.pipeline(
                self.hubert_model, self.net_g, sid, mono_for_pipeline, input_audio_path, times,
                f0_up_key, f0_method, file_index, index_rate, self.if_f0,
                filter_radius, self.tgt_sr, sr_rvc, rms_mix_rate, self.version, protect, f0_file
            )

            # Convert to float[-1,1] if it's int16
            if audio_opt_16k.dtype == np.int16:
                audio_opt_16k = audio_opt_16k.astype(np.float32) / 32768.0

            # ----------------------------------------------------------------
            # (E) Resample pipeline output back to original sr (og_sr)
            # ----------------------------------------------------------------
            if sr_rvc != og_sr:
                audio_opt_float = librosa.resample(audio_opt_16k, orig_sr=sr_rvc, target_sr=og_sr)
            else:
                audio_opt_float = audio_opt_16k

            # ----------------------------------------------------------------
            # (F) Reconstruct Stereo if applicable (Mid-Side Decoding)
            # ----------------------------------------------------------------
            if side_og is not None:
                if callback is not None:
                    callback(self.global_step / self.total_steps, "Reconstructing stereo (Mid-Side Decoding)",
                             self.total_steps)
                logger.info("Reconstructing stereo (Mid-Side Decoding)")
                new_len = len(audio_opt_float)
                if new_len != orig_len:
                    logger.info(f"Resampling side channel from {orig_len} to {new_len}")
                    side_og_resampled = resample_side(side_og, orig_len, new_len)
                else:
                    side_og_resampled = side_og
                logger.info(f"Resampled side channel shape: {side_og_resampled.shape}, converting.")
                output_stereo = mono_to_stereo_ms(audio_opt_float, side_og_resampled)
                final_float = output_stereo
            else:
                logger.info("Output is mono, no need to reconstruct stereo.")
                final_float = audio_opt_float.reshape(-1, 1)  # Ensure 2D shape

            # ----------------------------------------------------------------
            # (G) Apply Noise Removal (if enabled)
            # ----------------------------------------------------------------
            # if do_noise_removal:
            #     if callback is not None:
            #         callback(self.global_step / self.total_steps, "Applying noise removal...", self.total_steps)
            #     logger.info("Applying noise removal...")
            #     from handlers.noise_removal import restore_silence
            #     # Process each channel independently to ensure proper shape
            #     if final_float.ndim == 1 or final_float.shape[1] == 1:
            #         final_float = restore_silence(mono_og, final_float.flatten()).reshape(-1, 1)
            #     else:
            #         channels = []
            #         for ch in range(final_float.shape[1]):
            #             proc_channel = restore_silence(mono_og, final_float[:, ch])
            #             channels.append(proc_channel.reshape(-1, 1))
            #         final_float = np.concatenate(channels, axis=1)

            # ----------------------------------------------------------------
            # (I) Final Amplitude Scaling & Clipping
            # ----------------------------------------------------------------
            # max_val = np.max(np.abs(final_float))
            # if max_val > 1.0:
            #     logger.info(f"Clipping audio with max value: {max_val}")
            #     final_float /= max_val  # Simple limiting
            # final_float = np.clip(final_float, -1.0, 1.0)

            return f"Success. Processing time: {times}", (og_sr, final_float)

        except Exception as e:
            info = traceback.format_exc()
            logger.info("[vc_single ERROR]", info)
            return info, (None, None)

    # This is our entry point for the VC process
    def vc_multi(
            self,
            model,
            sid,
            paths,
            f0_up_key,
            f0_method,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            format1,
            project_dir,
            callback=None,
    ):
        outputs = []

        try:
            opt_root = os.path.join(project_dir, "cloned")
            os.makedirs(opt_root, exist_ok=True)
            first_path = paths[0]
            if not isinstance(first_path, str):
                paths = [p.name for p in paths]
            infos = []
            # Run self.get_vc to load the model if self.pipeline is None
            if self.pipeline is None:
                self.get_vc(model)
            for path in paths:
                if callback is not None:
                    callback(self.global_step / self.total_steps, f"Processing {path}", self.total_steps)
                info, opt = self.vc_single(
                    model,
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    callback=callback
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        # Extract base name and append "(Cloned)"
                        base_name, ext = os.path.splitext(os.path.basename(path))
                        cloned_name = f"{base_name}(Cloned).wav"

                        output_file = os.path.join(opt_root, f"{cloned_name}")
                        # Save the processed audio
                        sf.write(output_file, audio_opt, tgt_sr, format="wav", subtype="PCM_16")
                        outputs.append(output_file)
                        self.global_step += 1
                    except Exception as e:
                        traceback.print_exc()

        except:
            traceback.print_exc()
            logger.warning(traceback.format_exc())
        return outputs
