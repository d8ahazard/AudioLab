import logging
import os
import traceback
from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal

from handlers.config import model_path
from handlers.noise_removal import restore_silence
from handlers.stereo import stereo_to_mono_ms, resample_side, mono_to_stereo_ms

logger = logging.getLogger(__name__)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


def check_faiss_index_file(file_path):
    """
    Checks whether the given file appears to be a raw FAISS index file or a ZIP archive.

    Parameters:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file appears to be a valid FAISS index (non-ZIP), False otherwise.
    """
    import os

    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    with open(file_path, "rb") as f:
        header = f.read(4)
    if header == b"PK\x03\x04":
        print("The file appears to be a ZIP archive. Please unzip it to obtain the FAISS index file.")
        return False
    else:
        print("The file appears to be a valid FAISS index file (based on header check).")
        return True


def extract_index_from_zip(zip_path, extract_to):
    """
    Extracts files from a ZIP archive and returns the path to the first extracted file.

    Parameters:
        zip_path (str): Path to the ZIP archive.
        extract_to (str): Directory where the contents should be extracted.

    Returns:
        str: Path to the extracted index file.

    Raises:
        ValueError: If the file is not a valid ZIP archive or if extraction yields no files.
    """
    import os
    import zipfile

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid ZIP file.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    extracted_files = [f for f in os.listdir(extract_to) if os.path.isfile(os.path.join(extract_to, f))]
    if not extracted_files:
        raise ValueError("No files were extracted from the ZIP archive.")

    # Assumes that the first extracted file is the FAISS index file
    extracted_file_path = os.path.join(extract_to, extracted_files[0])
    print(f"Extracted file: {extracted_file_path}")
    return extracted_file_path


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        # Minimal chunking tweaks: add +0.2s to x_pad, +2s to x_query
        self.x_pad = config.x_pad + 0.2
        self.x_query = config.x_query + 2
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half

        self.sr = 16000
        self.window = 160
        self.t_pad = int(self.sr * self.x_pad)
        self.t_pad_tgt = int(tgt_sr * self.x_pad)
        self.t_pad2 = self.t_pad * 2
        self.t_query = int(self.sr * self.x_query)
        self.t_center = int(self.sr * self.x_center)
        self.t_max = int(self.sr * self.x_max)
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

    def vc(
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

    def _run_pipeline_mono(
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
            f0_file,
    ):
        # Helper function to print peak volume in dB
        def _print_peak_volume_db(signal, step_name):
            try:
                # Add a tiny offset in case signal is completely silent to avoid log10(0)
                peak = np.max(np.abs(signal)) + 1e-10
                db_val = 20.0 * np.log10(peak)
                logger.info(f"{step_name} peak volume: {db_val:.2f} dB")
            except Exception as e:
                pass

        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
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
        _print_peak_volume_db(audio, "After filtfilt")

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
            _print_peak_volume_db(seg_base, f"Before vc seg chunk #{idx}")
            if if_f0 == 1:
                seg = self.vc(
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
                seg = self.vc(
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
            _print_peak_volume_db(seg, f"After vc seg chunk #{idx}")

            audio_opt.append(seg)
            s = t

        # 3) Final chunk
        if if_f0 == 1:
            final_seg = self.vc(
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
            final_seg = self.vc(
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
        _print_peak_volume_db(final_seg, "After vc final seg")

        audio_opt.append(final_seg)
        audio_opt = np.concatenate(audio_opt)
        _print_peak_volume_db(audio_opt, "After concatenating all segments")

        # 4) Possibly change RMS
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
            _print_peak_volume_db(audio_opt, "After change_rms")

        # 5) Restore silence if needed
        if len(audio_opt) != len(og_audio):
            og_audio_resampled = librosa.resample(
                og_audio, orig_sr=self.sr, target_sr=(len(audio_opt) / len(og_audio)) * self.sr
            )
            logger.info(f"Resampled OG audio to match output length: {len(og_audio_resampled)}")
            audio_opt = restore_silence(og_audio_resampled, audio_opt)
            _print_peak_volume_db(audio_opt, "After restore_silence (resampled OG)")
        else:
            audio_opt = restore_silence(og_audio, audio_opt)
            _print_peak_volume_db(audio_opt, "After restore_silence")

        # 6) Resample to final sample rate if needed
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
            _print_peak_volume_db(audio_opt, "After final resample")

        # 7) Convert to int16
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            logger.info("Warning: audio_max > 1")
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        _print_peak_volume_db(audio_opt, "After float->int16 conversion")

        del pitch, pitchf, sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt

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
        """
        Processes the audio pipeline with proper RMS consistency, stereo handling,
        and integration of dynamic sample rate restoration.
        """
        if audio.ndim == 2:  # Stereo
            if audio.shape[0] == 2 and audio.shape[0] < audio.shape[1]:
                audio = audio.T

            if audio.shape[1] == 2:
                mono, side, orig_len = stereo_to_mono_ms(audio)
                orig_rms = np.sqrt(np.mean(mono ** 2)) + 1e-8

                processed_mono = self._run_pipeline_mono(
                    model, net_g, sid, mono, input_audio_path, times,
                    f0_up_key, f0_method, file_index, index_rate, if_f0,
                    filter_radius, tgt_sr, resample_sr, rms_mix_rate,
                    version, protect, f0_file
                )

                if processed_mono.dtype == np.int16:
                    processed_mono = processed_mono.astype(np.float32) / 32768.0

                proc_rms = np.sqrt(np.mean(processed_mono ** 2)) + 1e-8
                amp_factor = proc_rms / orig_rms
                side = resample_side(side, orig_len, len(processed_mono)) * amp_factor

                output_stereo = mono_to_stereo_ms(processed_mono, side)
                return output_stereo
            else:
                return self._run_pipeline_mono(
                    model, net_g, sid, audio.flatten(), input_audio_path, times,
                    f0_up_key, f0_method, file_index, index_rate, if_f0,
                    filter_radius, tgt_sr, resample_sr, rms_mix_rate,
                    version, protect, f0_file
                )
        else:  # Mono
            return self._run_pipeline_mono(
                model, net_g, sid, audio, input_audio_path, times,
                f0_up_key, f0_method, file_index, index_rate, if_f0,
                filter_radius, tgt_sr, resample_sr, rms_mix_rate,
                version, protect, f0_file
            )
