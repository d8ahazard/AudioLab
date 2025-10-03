import json
import logging
import os
import sys

if os.path.basename(sys.argv[0]) == "pipeline.py":
    # Set logging level to INFO
    logging.basicConfig(level=logging.INFO)
    # Disable fairseq, torch, faiss logging
    logging.getLogger("fairseq").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("faiss").setLevel(logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
import traceback
from time import time as ttime
import argparse
from typing import Dict, List, Optional, Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal

from handlers.autotune import auto_tune_track
from handlers.config import model_path, output_path
from handlers.spectrogram import F0Visualizer
from modules.rvc.infer.lib.audio import load_audio_advanced
from modules.rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from modules.rvc.infer.modules.vc.utils import (
    get_index_path_from_model,
    load_hubert,
)
from util.audio_track import shift_pitch

logger = logging.getLogger(__name__)
input_audio_path2wav = {}

# Global debug cloning settings
DEBUG_CLONE = False
DEBUG_STEP_NO = 0
DEBUG_TEST_ID = ""  # For identifying different test runs
DEBUG_TEST_KEY = ""

# Test parameter variations for experimentation
TEST_PARAMS = {
    "pitch_extraction_method": ["rmvpe+", "harvest", "hybrid"],
}

# Default parameters from the UI
DEFAULT_PARAMS = {
    "speaker_id": 0,
    "pitch_extraction_method": "harvest",
    "volume_mix_rate": 0.9,
    "protect": 0.2,
    "filter_radius": 3,
    "index_rate": 1.0,
    "merge_type": "mean",
    "crepe_hop_length": 160,
    "f0_autotune": False,
    "rmvpe_onnx": False,
    "clone_stereo": False,
    "pitch_correction": False,
    "pitch_correction_humanize": 0.95,
    "f0_up_key": 0
}


def debug_clone_audio(audio_data, sr, step_name):
    global DEBUG_STEP_NO, DEBUG_TEST_ID
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
    if DEBUG_TEST_ID:
        debug_folder = os.path.join(debug_folder, DEBUG_TEST_ID)
    os.makedirs(debug_folder, exist_ok=True)
    file_path = os.path.join(debug_folder, f"{step_name}_{DEBUG_TEST_KEY}.wav")
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
                       If None, then we force it to 16000 (Hubert expects 16kHz).
        """
        self.x_pad = config.x_pad + 0.2
        self.x_query = config.x_query + 2
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        # Store config and target SR for later use (e.g. for FeatureExtractor)
        self.config = config
        self.tgt_sr = tgt_sr

        # Force processing sample rate to 16000 (Hubert's magic value)
        if processing_sr is not None:
            self.sr = processing_sr
        else:
            self.sr = 16000

        self.window = 160
        self.t_pad = int(self.sr * self.x_pad)
        self.t_pad_tgt = int(tgt_sr * self.x_pad)
        self.t_pad2 = self.t_pad * 2
        self.t_query = int(self.sr * self.x_query)
        self.t_center = int(self.sr * self.x_center)
        self.t_max = int(self.sr * self.x_max)
        self.device = config.device
        # High-pass filter (5th order Butterworth at ~48 Hz) for input audio
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.tgt_sr = tgt_sr
        self.is_half = config.is_half
        # Disable post-pipeline hiss suppressor; we handle hiss at the mono stage in vc_single
        self.enable_post_hiss_notch = False

    def get_f0(self, audio, p_len, f0_up_key, f0_method, filter_radius, inp_f0=None,
               merge_type="median", crepe_hop_length=160, f0_autotune=False,
               rmvpe_onnx=False, f0_min=50, f0_max=1100):
        """Extract F0 using specified method, with optional smoothing."""
        from modules.rvc.pitch_extraction import FeatureExtractor
        fe = FeatureExtractor(self.tgt_sr, self.config, onnx=rmvpe_onnx)
        pitch, pitchf = fe.get_f0(
            audio, f0_up_key, f0_method,
            merge_type=merge_type,
            filter_radius=filter_radius,
            crepe_hop_length=crepe_hop_length,
            f0_autotune=f0_autotune,
            rmvpe_onnx=rmvpe_onnx,
            inp_f0=inp_f0,
            f0_min=f0_min, f0_max=f0_max
        )
        del fe
        # Ensure length consistency
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        # Apply median smoothing to F0 if filter_radius > 2
        if filter_radius is not None and filter_radius > 2:
            try:
                pitchf = signal.medfilt(pitchf, kernel_size=3)
            except Exception:
                pass
        if "mps" not in str(self.device) or "xpu" not in str(self.device):
            pitchf = pitchf.astype(np.float32)
        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        return pitch, pitchf

    def voice_clone(self, model, net_g, sid, audio_seg, pitch, pitchf,
                    times, index, big_npy, index_rate, version, protect):
        """Process a single audio segment through the model."""
        feats = torch.from_numpy(audio_seg)
        feats = feats.half() if self.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1).to(self.device)
        padding_mask = torch.zeros_like(feats, dtype=torch.bool, device=self.device)
        inputs = {"source": feats, "padding_mask": padding_mask,
                  "output_layer": 9 if version == "v1" else 12}
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
            feats0 = torch.nn.functional.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t1 = ttime()
        seg_frames = audio_seg.shape[0] // self.window
        if feats.shape[1] < seg_frames:
            seg_frames = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :seg_frames]
                pitchf = pitchf[:, :seg_frames]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            mask = pitchf.clone()
            mask[pitchf > 0] = 1.0
            mask[pitchf < 1] = protect
            mask = mask.unsqueeze(-1)
            feats = feats * mask + feats0 * (1 - mask)
            feats = feats.to(feats0.dtype)
        seg_len_tensor = torch.tensor([seg_frames], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio_out = net_g.infer(feats, seg_len_tensor, pitch, pitchf, sid)[0][0,0].data.cpu().float().numpy()
            else:
                audio_out = net_g.infer(feats, seg_len_tensor, sid)[0][0,0].data.cpu().float().numpy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio_out

    def pipeline(self, model, net_g, sid, audio, times,
                 f0_up_key, f0_method, file_index, index_rate,
                 if_f0, filter_radius, merge_type, crepe_hop_length,
                 f0_autotune, rmvpe_onnx, tgt_sr, og_sr,
                 rms_mix_rate, version, protect, pitch_correction,
                 pitch_correction_humanize, f0_file=None):
        index = big_npy = None
        if file_index is not None and file_index != "" and index_rate != 0 and os.path.exists(file_index):
            try:
                import faiss
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                traceback.print_exc()
                index = big_npy = None
        try:
            audio = signal.filtfilt(self.bh, self.ah, audio)
        except Exception as e:
            pass
        audio_pad = np.pad(audio, (self.window//2, self.window//2), mode="reflect")
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
        audio_opt_segments = []
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        inp_f0 = None
        if f0_file is not None and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = [line.strip() for line in f]
                inp_f0 = np.array([[float(x) for x in line.split(",")] for line in lines], dtype="float32")
            except Exception as e:
                traceback.print_exc()
                inp_f0 = None
        sid_tensor = torch.tensor([sid], device=self.device).long()
        pitch = pitchf = None
        t_start = ttime()
        p_len = audio_pad.shape[0] // self.window
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method,
                                        filter_radius, inp_f0,
                                        merge_type=merge_type,
                                        crepe_hop_length=crepe_hop_length,
                                        f0_autotune=f0_autotune, rmvpe_onnx=rmvpe_onnx)
        t_end = ttime()
        times[1] += t_end - t_start
        for t in opt_ts:
            t = (t // self.window) * self.window
            segment = audio_pad[s : t + self.t_pad2 + self.window]
            if if_f0 == 1:
                seg_pitch   = pitch[:,  s//self.window : (t + self.t_pad2)//self.window]
                seg_pitchf  = pitchf[:, s//self.window : (t + self.t_pad2)//self.window]
            else:
                seg_pitch = seg_pitchf = None
            seg_audio = self.voice_clone(model, net_g, sid_tensor, segment,
                                         seg_pitch, seg_pitchf, times,
                                         index, big_npy, index_rate,
                                         version, protect)
            seg_audio = seg_audio[self.t_pad_tgt : -self.t_pad_tgt]
            audio_opt_segments.append(seg_audio)
            s = t
        final_segment = audio_pad[s:]
        if if_f0 == 1:
            final_pitch  = pitch[:,  s//self.window:] if s else pitch
            final_pitchf = pitchf[:, s//self.window:] if s else pitchf
        else:
            final_pitch = final_pitchf = None
        final_audio = self.voice_clone(model, net_g, sid_tensor, final_segment,
                                       final_pitch, final_pitchf, times,
                                       index, big_npy, index_rate,
                                       version, protect)
        final_audio = final_audio[self.t_pad_tgt : -self.t_pad_tgt]
        audio_opt_segments.append(final_audio)
        audio_opt = np.concatenate(audio_opt_segments)
        debug_clone_audio(audio_opt, tgt_sr, f"vc_single_final_audio")
        if pitch_correction:
            audio_opt, detected_key, scale = auto_tune_track(
                audio_opt, tgt_sr, strength=0.5,
                humanize=pitch_correction_humanize, f0_method=f0_method
            )
            debug_clone_audio(audio_opt, tgt_sr, f"vc_single_final_audio_after_pitch_correction")
        if tgt_sr != og_sr and og_sr is not None:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=og_sr)
            final_sr = og_sr
            debug_clone_audio(audio_opt, final_sr, f"vc_single_final_audio_after_resampling")
        else:
            final_sr = tgt_sr

        peak = np.max(np.abs(audio_opt))
        if peak > 0.99:
            audio_opt = audio_opt * (0.99 / peak)
            debug_clone_audio(audio_opt, final_sr, f"vc_single_final_audio_after_peak_clipping")
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
        self.cpt = torch.load(person, map_location="cpu", weights_only=True)
        self.tgt_sr = self.cpt["config"][-1]
        logger.info(f"Target SR from checkpoint: {self.tgt_sr}")
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
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
        if not self.downsample_pipeline:
            self.pipeline = Pipeline(self.tgt_sr, self.config, self.downsample_pipeline,
                                     processing_sr=None)
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

    def vc_single(self, model, sid, input_audio_path,
                  f0_up_key=0, f0_file=None, f0_method="crepe",
                  index_rate=0.0, filter_radius=3, rms_mix_rate=1.0,
                  protect=0.33, pitch_correction=False, pitch_correction_humanize=False,
                  merge_type="median", crepe_hop_length=160, f0_autotune=False,
                  rmvpe_onnx=False, clone_stereo=False, callback=None):

        """
        Run voice conversion on a single audio file.
        """
        audio_float, og_sr = load_audio_advanced(file=input_audio_path, sr=None, mono=False, return_sr=True)
        # Report load immediately
        if callback is not None:
            callback(
                self.global_step / self.total_steps,
                f"Loaded audio: shape={audio_float.shape}, original SR={og_sr}",
                self.total_steps
            )
        # Apply pre-clone pitch shift if requested
        if f0_up_key != 0:
            try:
                logger.info(f"[RVC] Applying pitch shift of {f0_up_key} semitones before cloning")
                audio_float, og_sr = shift_pitch((audio_float, og_sr), f0_up_key)
                f0_up_key = 0  # ensure no further pitch shifting downstream
            except Exception as e:
                logger.warning(f"[RVC] Pitch shift failed, proceeding without shift: {e}")
        try:
            sr_rvc = 16000  # Always process at 16kHz for Hubert
            if not self.downsample_pipeline:
                self.pipeline = Pipeline(self.tgt_sr, self.config, self.downsample_pipeline, processing_sr=og_sr)
            if self.pipeline is None:
                self.get_vc(model)
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)
            # Hiss suppressor removed per request (muffled vocals); keeping stub NO-OP
            def _apply_mono_hiss_notch(orig_mono: np.ndarray, cloned_mono: np.ndarray, sr: int) -> np.ndarray:
                return cloned_mono
            def process_track(wav, label):
                if og_sr != sr_rvc:
                    wav_resamp = librosa.resample(wav, orig_sr=og_sr, target_sr=sr_rvc)
                else:
                    wav_resamp = wav

                debug_clone_audio(wav_resamp, sr_rvc, f"vc_single_{label}_pre_pipeline")
                if callback is not None:
                    callback(
                        self.global_step / self.total_steps,
                        f"Running {label} pipeline...",
                        self.total_steps
                    )

                out_wav, proc_sr = self.pipeline.pipeline(
                    self.hubert_model, self.net_g, sid, wav_resamp, [0, 0, 0],
                    f0_up_key, f0_method, self.index, index_rate, self.if_f0,
                    filter_radius, merge_type, crepe_hop_length, f0_autotune, rmvpe_onnx,
                    self.tgt_sr, og_sr, rms_mix_rate, self.version, protect,
                    pitch_correction, pitch_correction_humanize, f0_file
                )
                debug_clone_audio(out_wav, proc_sr, f"vc_single_{label}_post_pipeline")
                return out_wav, proc_sr
            if audio_float.ndim == 2 and audio_float.shape[1] == 2:
                left_channel = audio_float[:, 0]
                right_channel = audio_float[:, 1]
                if clone_stereo:
                    # Process left and right channels separately for full cloning
                    left_conv, proc_sr = process_track(left_channel, "left")
                    right_conv, proc_sr = process_track(right_channel, "right")
                    min_len = min(len(left_conv), len(right_conv))
                    final_audio = np.stack([left_conv[:min_len], right_conv[:min_len]], axis=1)
                else:
                    mono = 0.5 * (left_channel + right_channel)
                    mono_conv, proc_sr = process_track(mono, "mono")
                    # Hiss notch disabled
                    # Match RMS post-filter
                    orig_rms = np.sqrt(np.mean(mono**2))
                    proc_rms = np.sqrt(np.mean(mono_conv**2)) or 1e-8
                    mono_conv *= (orig_rms / proc_rms)
                    final_audio = np.stack([mono_conv, mono_conv], axis=1)
            else:
                mono = audio_float if audio_float.ndim == 1 else audio_float.mean(axis=1)
                mono_conv, proc_sr = process_track(mono, "mono")
                # Hiss notch disabled
                # Match RMS to original mono and output dual-mono to avoid channel discrepancies
                orig_rms = np.sqrt(np.mean(mono**2))
                proc_rms = np.sqrt(np.mean(mono_conv**2)) or 1e-8
                mono_conv *= (orig_rms / proc_rms)
                final_audio = np.stack([mono_conv, mono_conv], axis=1)
            # debug_clone_audio(final_audio, proc_sr, "vc_single_final_audio_before_silence_restore")
            # final_audio = restore_silence(audio_float, final_audio, og_sr, proc_sr)
            debug_clone_audio(final_audio, proc_sr, "vc_single_final_audio")
            final_float = final_audio
            visualizer = F0Visualizer()
            output_file = os.path.join(output_path, "spec.png")
            visualizer.visualize(output_file, sr=proc_sr, hop_length=crepe_hop_length)
            logger.info(f"Visualized: {output_file}")
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
            merge_type,
            crepe_hop_length,
            f0_autotune,
            rmvpe_onnx,
            clone_stereo,
            pitch_correction,
            pitch_correction_humanize,
            project_dir,
            callback=None,
    ):
        clone_params = {
            "model": model,
            "sid": sid,
            "f0_up_key": f0_up_key,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "filter_radius": filter_radius,
            "rms_mix_rate": rms_mix_rate,
            "protect": protect,
            "merge_type": merge_type,
            "crepe_hop_length": crepe_hop_length,
            "f0_autotune": f0_autotune,
            "rmvpe_onnx": rmvpe_onnx,
            "clone_stereo": clone_stereo,
            "pitch_correction": pitch_correction,
            "pitch_correction_humanize": pitch_correction_humanize
        }
        outputs = []
        global DEBUG_STEP_NO
        DEBUG_STEP_NO = 0
        try:
            opt_root = os.path.join(project_dir, "cloned")
            os.makedirs(opt_root, exist_ok=True)
            if not isinstance(paths[0], str):
                paths = [p.name for p in paths]
            if self.pipeline is None:
                self.get_vc(model)
            clone_params_file = os.path.join(opt_root, "clone_params.json")

            for path in paths:
                if callback is not None:
                    callback(self.global_step / self.total_steps,
                             f"Processing {os.path.basename(path)}", self.total_steps)

                base_name, ext = os.path.splitext(os.path.basename(path))
                model_base, _ = os.path.splitext(os.path.basename(model))
                # Append pitch shift to filename if non-zero: e.g., (-1), (+2), (+0.5)
                pitch_suffix = ""
                try:
                    if isinstance(f0_up_key, (int, float)) and f0_up_key != 0:
                        # Show sign with one decimal if needed, trim trailing .0
                        val = float(f0_up_key)
                        sign = "+" if val > 0 else "-"
                        abs_val = abs(val)
                        text_val = ("%g" % abs_val)
                        pitch_suffix = f"({sign}{text_val})"
                except Exception:
                    pitch_suffix = ""

                cloned_name = f"{base_name}(Cloned)({model_base}_{f0_method}){pitch_suffix}.wav"
                output_file = os.path.join(opt_root, cloned_name)
                param_match = False
                if os.path.exists(clone_params_file):
                    with open(clone_params_file, "r") as f:
                        clone_params_loaded = json.load(f)
                    param_match = clone_params_loaded == clone_params
                if os.path.exists(output_file) and param_match:
                    logger.info(f"Skipping {path} as {output_file} already exists.")
                    outputs.append(output_file)
                    continue

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
                    pitch_correction,
                    pitch_correction_humanize,
                    merge_type,
                    crepe_hop_length,
                    f0_autotune,
                    rmvpe_onnx,
                    clone_stereo,
                    callback=callback,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        sf.write(output_file, audio_opt, tgt_sr, format="wav", subtype="PCM_16")
                        outputs.append(output_file)
                    except Exception as e:
                        traceback.print_exc()
                else:
                    logger.error(f"Failed to process {path}: {info}")
                    raise Exception(f"Failed to process {path}: {info}")

            with open(clone_params_file, "w") as f:
                json.dump(clone_params, f, indent=4)

            return outputs
        except Exception:
            traceback.print_exc()
            logger.warning(traceback.format_exc())
            raise Exception("Failed to process audio files.")

def test_clone(input_path: str, model_name: str, test_params: Optional[Dict[str, List[Any]]] = None, 
               run_id: Optional[str] = None) -> None:
    """
    Test the voice cloning pipeline with different parameter combinations.
    
    Args:
        input_path: Path to the input audio file
        model_name: Name of the voice model to use (without .pth extension)
        test_params: Dictionary of parameters to test with their values
        run_id: Optional identifier for the test run
    """
    global DEBUG_CLONE, DEBUG_TEST_ID, DEBUG_STEP_NO, DEBUG_TEST_KEY
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
    DEBUG_CLONE = False
    base_test_id = run_id or f"test_{int(ttime())}"
    DEBUG_TEST_ID = base_test_id
    DEBUG_TEST_KEY = base_test_id
    DEBUG_STEP_NO = 0
    
    from modules.rvc.configs.config import Config
    config = Config()
    vc = VC(config, False)
    
    model_file = os.path.join(model_path, "trained", f"{model_name}.pth")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    if test_params is None:
        params = DEFAULT_PARAMS.copy()
        vc.get_vc(model_file)
        
        try:
            info, opt = vc.vc_single(
                model=model_file,
                sid=params["speaker_id"],
                input_audio_path=input_path,
                f0_up_key=params["f0_up_key"],
                f0_file=None,
                f0_method=params["pitch_extraction_method"],
                index_rate=params["index_rate"],
                filter_radius=params["filter_radius"],
                rms_mix_rate=params["volume_mix_rate"],
                protect=params["protect"],
                pitch_correction=params["pitch_correction"],
                pitch_correction_humanize=params["pitch_correction_humanize"],
                merge_type=params["merge_type"],
                crepe_hop_length=params["crepe_hop_length"],
                f0_autotune=params["f0_autotune"],
                rmvpe_onnx=params["rmvpe_onnx"],
                clone_stereo=params["clone_stereo"]
            )
            
            if "Success" in info:
                output_dir = os.path.join(output_path, "test_clone")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"test_clone_default_{DEBUG_TEST_ID}.wav")
                tgt_sr, audio_opt = opt
                sf.write(output_file, audio_opt, tgt_sr)
                print(f"Default test output saved to: {output_file}")
            
        except Exception as e:
            print(f"Error during default test: {str(e)}")
            traceback.print_exc()
            
    else:
        for param_name, param_values in test_params.items():
            if param_name not in DEFAULT_PARAMS:
                print(f"Warning: Unknown parameter {param_name}, skipping...")
                continue
                
            for value in param_values:
                DEBUG_STEP_NO = 0
                DEBUG_TEST_KEY = f"{param_name}_{value}"
                
                params = DEFAULT_PARAMS.copy()
                params[param_name] = value
                
                try:
                    vc.get_vc(model_file)
                    
                    info, opt = vc.vc_single(
                        model=model_file,
                        sid=params["speaker_id"],
                        input_audio_path=input_path,
                        f0_up_key=params["f0_up_key"],
                        f0_file=None,
                        f0_method=params["pitch_extraction_method"],
                        index_rate=params["index_rate"],
                        filter_radius=params["filter_radius"],
                        rms_mix_rate=params["volume_mix_rate"],
                        protect=params["protect"],
                        pitch_correction=params["pitch_correction"],
                        pitch_correction_humanize=params["pitch_correction_humanize"],
                        merge_type=params["merge_type"],
                        crepe_hop_length=params["crepe_hop_length"],
                        f0_autotune=params["f0_autotune"],
                        rmvpe_onnx=params["rmvpe_onnx"],
                        clone_stereo=params["clone_stereo"]
                    )
                    
                    if "Success" in info:
                        output_dir = os.path.join(output_path, "test_clone")
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, 
                            f"test_clone_{param_name}_{value}_{base_test_id}.wav")
                        tgt_sr, audio_opt = opt
                        sf.write(output_file, audio_opt, tgt_sr)
                        print(f"Test output saved for {param_name}={value}: {output_file}")
                        
                except Exception as e:
                    print(f"Error testing {param_name}={value}: {str(e)}")
                    traceback.print_exc()

if __name__ == "__main__":
    if os.path.basename(sys.argv[0]) == "pipeline.py":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
    parser = argparse.ArgumentParser(description="Test RVC voice cloning with different parameters")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("model_name", help="Name of the voice model to use (without .pth extension)")
    parser.add_argument("--run-id", help="Optional identifier for the test run", default=None)
    parser.add_argument("--param", help="Parameter to test (if not specified, uses defaults)", default=None)
    args = parser.parse_args()
    
    if args.param and args.param in TEST_PARAMS:
        test_params = {args.param: TEST_PARAMS[args.param]}
    else:
        test_params = TEST_PARAMS
        
    test_clone(args.input_path, args.model_name, test_params, args.run_id)
