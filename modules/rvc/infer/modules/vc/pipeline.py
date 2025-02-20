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

from handlers.config import model_path, output_path
from handlers.noise_removal import restore_silence
from handlers.stereo import stereo_to_mono_ms, resample_side, mono_to_stereo_ms
from modules.rvc.infer.lib.audio import load_audio_advanced
from modules.rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from modules.rvc.infer.modules.vc.utils import (
    check_faiss_index_file,
    extract_index_from_zip,
    cache_harvest_f0,
    change_rms,
    get_index_path_from_model,
    load_hubert,
)
from util.audio_track import shift_pitch

logger = logging.getLogger(__name__)
input_audio_path2wav = {}

# Global debug cloning settings
DEBUG_CLONE = True
DEBUG_STEP_NO = 0


def debug_clone_audio(audio_data, sr, step_name):
    global DEBUG_STEP_NO
    """
    Logs the max amplitude, RMS, sample rate, and length of the audio data,
    and saves the audio file to disk with the step name (prefixed for ordering).
    """
    max_amp = np.max(np.abs(audio_data)) + 1e-10
    rms = np.sqrt(np.mean(audio_data ** 2))
    length = len(audio_data)
    step_name = f"{DEBUG_STEP_NO:03d}_{step_name}"
    step_name_len = len(step_name)
    padding = " " * (30 - step_name_len) if step_name_len < 30 else ""
    logger.info(f"{step_name}:{padding}SR={sr}, Length={length}, Max Amp={max_amp:.4f}, RMS={rms:.4f}")

    if not DEBUG_CLONE:
        return
    debug_folder = os.path.join(output_path, "debug")
    os.makedirs(debug_folder, exist_ok=True)
    file_path = os.path.join(debug_folder, f"{step_name}.wav")
    try:
        sf.write(file_path, audio_data, sr)
    except Exception as e:
        logger.error(f"[DEBUG_CLONE] Failed to save debug audio {file_path}: {e}")
    DEBUG_STEP_NO += 1


class Pipeline(object):
    def __init__(self, tgt_sr, config, downsample_pipeline, processing_sr=None):
        """
        tgt_sr: target sample rate from the model checkpoint (e.g. 48000)
        config: configuration object containing parameters including:
            - x_pad, x_query, x_center, x_max, is_half, downsample_pipeline (bool)
        processing_sr: if provided, the sample rate to use for all pipeline processing.
                       If None, then it is set to 16000 when downsampling is enabled,
                       or to tgt_sr when disabled.
        """
        self.x_pad = config.x_pad + 0.2
        self.x_query = config.x_query + 2
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        # Decide on the processing sample rate:
        if processing_sr is not None:
            self.sr = processing_sr
        else:
            self.sr = 16000 if downsample_pipeline else tgt_sr
        # This was 160, but we're using 80 for now
        self.window = 160
        self.t_pad = int(self.sr * self.x_pad)
        self.t_pad_tgt = int(tgt_sr * self.x_pad)
        self.t_pad2 = self.t_pad * 2
        self.t_query = int(self.sr * self.x_query)
        self.t_center = int(self.sr * self.x_center)
        self.t_max = int(self.sr * self.x_max)
        self.device = config.device

        # Compute filter coefficients using self.sr instead of a fixed 16000.
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)

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
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
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
                self.model_rmvpe = RMVPE(
                    model_path=rvmpe_model_path,
                    is_half=self.is_half,
                    device=self.device,
                    sampling_rate=self.sr,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            if "privateuseone" in str(self.device):
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning runtime memory")

        # Pitch shift
        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window

        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = ((f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1)
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
        # Process the input segment (audio0) and perform voice cloning.
        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.zeros(feats.shape, dtype=torch.bool, device=self.device)
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

        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half:
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

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
            og_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=None,
    ):
        # Get FAISS index if available
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

        # 1) Filtering using filtfilt with coefficients computed at self.sr
        audio = signal.filtfilt(self.bh, self.ah, audio)
        debug_clone_audio(audio, self.sr, "filtfilt")

        # Prepare segmentation by padding
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
                inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
            except:
                traceback.print_exc()

        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path, audio_pad, p_len, f0_up_key, f0_method, filter_radius, inp_f0
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1

        # Process each segment except the final chunk
        for idx, t in enumerate(opt_ts):
            t = (t // self.window) * self.window
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
                    model, net_g, sid_tensor, seg_base, None, None, times, index, big_npy, index_rate, version, protect
                )
            seg = seg[self.t_pad_tgt: -self.t_pad_tgt]
            audio_opt.append(seg)
            s = t

        # Process final chunk
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
                model, net_g, sid_tensor, audio_pad[s:], None, None, times, index, big_npy, index_rate, version, protect
            )
        final_seg = final_seg[self.t_pad_tgt: -self.t_pad_tgt]
        audio_opt.append(final_seg)
        audio_opt = np.concatenate(audio_opt)
        audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=og_sr)

        debug_clone_audio(audio_opt, og_sr, "after_vc_concat")

        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, self.sr, audio_opt, tgt_sr, rms_mix_rate)
            debug_clone_audio(audio_opt, og_sr, "after_change_rms")

        final_sr = og_sr
        # debug_clone_audio(audio_opt, final_sr, "after_int16_conversion")

        del pitch, pitchf, sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt, final_sr


class VC:
    def __init__(self, config, downsample_pipeline):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline: Pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.hubert_model = None
        self.index = None
        self.config = config
        self.global_step = 0
        self.total_steps = 0
        self.downsample_pipeline = downsample_pipeline

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33),
            "__type__": "update",
        }
        if sid == "" or sid == []:
            if self.hubert_model is not None:
                logger.info("Cleaning model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)
                self.hubert_model = self.net_g = self.n_spk = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", None)
                if not self.version:
                    sval = self.cpt["config"][14][0]
                    self.version = "v2" if sval == 24 else "v1"
                logger.info(f"IF_F0: {self.if_f0}, Version: {self.version}")

                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(*self.cpt["config"], is_half=self.config.is_half)
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.config.is_half)
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {"visible": True, "value": to_return_protect1, "__type__": "update"},
            )
        person = os.path.join(model_path, "trained", sid)
        logger.info(f"Loading: {person}")
        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        logger.info(f"Target SR from checkpoint: {self.tgt_sr}")
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
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()
        # Instantiate Pipeline. If downsampling is enabled (default), use 16k;
        # otherwise, reinitialize the pipeline to process at the original input rate.
        if not self.downsample_pipeline:
            # Reinitialize Pipeline with processing_sr = original input rate.
            self.pipeline = Pipeline(self.tgt_sr, self.config, self.downsample_pipeline,
                                     processing_sr=None)  # will be re-created in vc_single
        else:
            self.pipeline = Pipeline(self.tgt_sr, self.config, self.downsample_pipeline)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(person), "__type__": "update"}
        logger.info(f"Select index: {index}")
        self.index = index["value"]
        return (
            ({"visible": True, "maximum": n_spk, "__type__": "update"}, to_return_protect1)
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

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
            rms_mix_rate,
            protect,
            clone_stereo=False,
            callback=None,
    ):
        try:
            # (A) Load original audio at its native sample rate.
            audio_float, og_sr = load_audio_advanced(
                file=input_audio_path, sr=None, mono=False, return_sr=True
            )
            if f0_up_key != 0:
                # Pitch shift audio_float and set f0_up_key to 0.
                audio_float, og_sr = shift_pitch((audio_float, og_sr), f0_up_key)
                f0_up_key = 0

            debug_clone_audio(audio_float, og_sr, "vc_single_loaded_audio")
            if callback is not None:
                callback(
                    self.global_step / self.total_steps,
                    f"Loaded audio: shape={audio_float.shape}, original SR={og_sr}",
                    self.total_steps
                )

            # (B) Determine processing sample rate.
            sr_rvc = 16000 if self.downsample_pipeline else og_sr
            if not self.downsample_pipeline:
                # Reinitialize the pipeline for original sample rate.
                self.pipeline = Pipeline(self.tgt_sr, self.config, self.downsample_pipeline, processing_sr=og_sr)

            # (C) Ensure our models are loaded.
            if self.pipeline is None:
                self.get_vc(model)
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            # (D) Helper function to process a single track.
            def process_track(track, label):
                # Resample if needed.
                if og_sr != sr_rvc:
                    track_resampled = librosa.resample(track, orig_sr=og_sr, target_sr=sr_rvc)
                else:
                    track_resampled = track

                debug_clone_audio(track_resampled, sr_rvc, f"vc_single_{label}_pre_pipeline")
                if callback is not None:
                    callback(
                        self.global_step / self.total_steps,
                        f"Running {label} pipeline...",
                        self.total_steps
                    )

                processed, proc_sr = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    track_resampled,
                    input_audio_path,
                    [0, 0, 0],  # times placeholder
                    f0_up_key,
                    f0_method,
                    self.index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr if self.downsample_pipeline else og_sr,
                    og_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    f0_file,
                )
                return processed, proc_sr

            # (E) Process based on input channel configuration.
            if audio_float.ndim == 2 and audio_float.shape[1] == 2:
                # Stereo input.
                left = audio_float[:, 0]
                right = audio_float[:, 1]

                if clone_stereo:
                    # --- Process only MID; keep SIDE unprocessed ---
                    mid = 0.5 * (left + right)
                    side = 0.5 * (left - right)
                    debug_clone_audio(mid, og_sr, "vc_single_mid_channel")

                    # Process just the MID channel
                    mid_opt, proc_sr = process_track(mid, "mid")

                    # Resample side channel if needed
                    if og_sr != proc_sr:
                        side_resampled = librosa.resample(side, orig_sr=og_sr, target_sr=proc_sr)
                    else:
                        side_resampled = side

                    # Trim both to the same length
                    min_len = min(len(mid_opt), len(side_resampled))
                    mid_opt = mid_opt[:min_len]
                    side_resampled = side_resampled[:min_len]

                    # --- Preserve original mid/side ratio ---
                    orig_mid_rms = max(1e-8, np.sqrt(np.mean(mid[:min_len] ** 2)))
                    proc_mid_rms = max(1e-8, np.sqrt(np.mean(mid_opt ** 2)))
                    volume_ratio = orig_mid_rms / proc_mid_rms

                    # Scale both processed mid and side by the same amount
                    mid_opt *= volume_ratio
                    side_resampled *= volume_ratio

                    # Reconstruct stereo
                    final_left = mid_opt + side_resampled
                    final_right = mid_opt - side_resampled
                    final_float = np.stack([final_left, final_right], axis=1)
                else:
                    # --- Non-clone stereo: convert to mono (keeping side info) and process.
                    mono_og, side_og, orig_len = stereo_to_mono_ms(audio_float)
                    debug_clone_audio(mono_og, og_sr, "vc_single_mono_converted")
                    processed_mono, proc_sr = process_track(mono_og, "mono")

                    # Reconstruct stereo using resampled side information
                    new_len = len(processed_mono)
                    if new_len != orig_len:
                        side_resampled = resample_side(side_og, orig_len, new_len)
                    else:
                        side_resampled = side_og
                    final_float = mono_to_stereo_ms(processed_mono, side_resampled)

                    # --- Volume adjustment: match the RMS of the original mono
                    original_rms = np.sqrt(np.mean(mono_og ** 2))
                    processed_rms = np.sqrt(np.mean(processed_mono ** 2))
                    if processed_rms > 0:
                        volume_adjust = original_rms / processed_rms
                    else:
                        volume_adjust = 1.0
                    final_float *= volume_adjust
            else:
                # Mono input
                if audio_float.ndim == 1:
                    mono = audio_float
                else:
                    mono = audio_float.mean(axis=1)
                debug_clone_audio(mono, og_sr, "vc_single_mono_ready")
                processed_mono, proc_sr = process_track(mono, "mono")
                final_float = processed_mono.reshape(-1, 1)

            # (F) Call silence restoration once
            #final_float = restore_silence(audio_float, final_float, og_sr, proc_sr)
            #debug_clone_audio(final_float, proc_sr, "vc_single_after_silence_restore")

            self.global_step += 1
            return f"Success. Processing time: {[0, 0, 0]}", (og_sr, final_float)
        except Exception as e:
            info = traceback.format_exc()
            logger.info("[vc_single ERROR] " + info)
            return info, (None, None)

    def vc_multi(
            self,
            model,
            sid,
            paths,
            f0_up_key,
            f0_method,
            index_rate,
            filter_radius,
            rms_mix_rate,
            protect,
            clone_stereo,
            project_dir,
            callback=None,
    ):
        outputs = []
        global DEBUG_STEP_NO
        DEBUG_STEP_NO = 0
        try:
            opt_root = os.path.join(project_dir, "cloned")
            os.makedirs(opt_root, exist_ok=True)
            if not isinstance(paths[0], str):
                paths = [p.name for p in paths]
            # Ensure model is loaded
            if self.pipeline is None:
                self.get_vc(model)
            for path in paths:
                if callback is not None:
                    callback(self.global_step / self.total_steps, f"Processing {os.path.basename(path)}", self.total_steps)
                info, opt = self.vc_single(
                    model,
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect,
                    clone_stereo,
                    callback=callback,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        base_name, ext = os.path.splitext(os.path.basename(path))
                        cloned_name = f"{base_name}(Cloned).wav"
                        output_file = os.path.join(opt_root, cloned_name)
                        sf.write(output_file, audio_opt, tgt_sr, format="wav", subtype="PCM_16")
                        outputs.append(output_file)
                    except Exception as e:
                        traceback.print_exc()
            return outputs
        except Exception:
            traceback.print_exc()
            logger.warning(traceback.format_exc())
            return outputs
