import logging
import os
from functools import lru_cache

import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe
from pyannote.audio import Pipeline as PyannotePipeline, Model
from scipy import signal

from handlers.config import model_path

logger = logging.getLogger(__name__)

# Global dictionary for caching audio for pyworld harvest
input_audio_path2wav = {}

# High-pass filter design: 5th order, cutoff ~48Hz at 16kHz
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    """
    Cache and return the F0 for the specified audio path.
    Uses pyworld's harvest + stonemask to refine pitch.
    """
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


def change_rms(
        data1: np.ndarray,
        sr1: int,
        data2: np.ndarray,
        sr2: int,
        mix_rate: float,
) -> np.ndarray:
    """
    Adjust RMS of data2 to partially match data1.
    'mix_rate' determines the extent:
      - 0.0 => data2 retains original RMS entirely,
      - 1.0 => data2 fully matches data1's RMS,
      - 0.5 => half-blend.
    """
    eps = 1e-6  # small epsilon to avoid division by zero

    # Calculate RMS energy for each half-second chunk
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    # Convert to torch tensors for interpolation
    rms1_t = torch.from_numpy(rms1).unsqueeze(0)
    rms2_t = torch.from_numpy(rms2).unsqueeze(0)

    # Match lengths via interpolation
    rms1_interp = F.interpolate(rms1_t, size=data2.shape[0], mode="linear").squeeze()
    rms2_interp = F.interpolate(rms2_t, size=data2.shape[0], mode="linear").squeeze()
    rms2_interp = torch.max(rms2_interp, torch.zeros_like(rms2_interp) + eps)

    # Mix the RMS values
    ratio = (
        torch.pow(rms1_interp, torch.tensor(1.0 - mix_rate))
        * torch.pow(rms2_interp, torch.tensor(mix_rate - 1.0))
    )
    data2 *= ratio.numpy()
    return data2


class Pipeline:
    """
    A pipeline for performing voice conversion tasks using RVC (Retrieval-based Voice Conversion).
    Handles chunking (now via Pyannote VAD), pitch extraction, RMS matching, and model inference.
    """

    def __init__(self, tgt_sr, config):
        # Basic chunking config
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max

        # Possibly set the high-pass filter globally (if you like):
        self.bh, self.ah = bh, ah

        self.is_half = config.is_half
        self.device = config.device

        # For RVC, input is resampled to 16k
        self.sr = 16000
        # self.window: number of audio samples per frame at 16k
        # 160 samples => 10ms frames
        self.window = 160

        # x_pad is in seconds in your config, so we convert to samples at 16k
        self.t_pad = self.sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max

        # For output chunking
        self.t_pad_tgt = tgt_sr * self.x_pad

        try:
            self.vad_model = Model.from_pretrained("pyannote/segmentation")
            self.vad_model.eval()
        except Exception as e:
            print(f"Could not load pyannote segmentation model: {e}")
            self.vad_model = None

        # RMVPE model placeholder (for "rmvpe" pitch extraction)
        self.model_rmvpe = None

    def amplitude_db(self, wave: np.ndarray) -> float:
        """Rough amplitude->dB measure for an entire chunk."""
        val = np.sqrt(np.mean(wave ** 2) + 1e-8)
        return 20.0 * np.log10(val)

    def voice_activity_segments(self, audio: np.ndarray, sr: int = 16000, threshold: float = 0.6):
        """
        Use pyannote segmentation model to approximate VAD by taking
        the maximum activation across classes and treating that as "speech probability."
        Return a list of (start_sample, end_sample).
        """
        import torch
        import numpy as np
        from pyannote.core import SlidingWindow

        if self.vad_model is None:
            print("Warning: self.vad_model is None. Returning single segment fallback.")
            return [(0, len(audio))]

        # Convert to float32 if needed & shape => (batch=1, n_samples)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # shape: (1, n_samples)

        with torch.no_grad():
            # segmentation shape => (num_frames, num_classes)
            segmentation = self.vad_model(waveform).cpu().numpy()[0]  # take batch idx 0

        # The pyannote "frames" property tells us the time step (in seconds) for each row in `segmentation`
        frames = self.vad_model.introspection.frames  # a SlidingWindow
        # frames.start = start time of first frame
        # frames.duration = duration of each frame
        # frames.step = step between frames

        # We'll treat speech_probability as the max activation across classes
        # => shape is (num_frames, 1)
        speech_probability = np.max(segmentation, axis=1, keepdims=True)

        # Now find contiguous regions where speech_probability > threshold
        # Convert each frame index => time in seconds => samples
        sr_per_frame = frames.step  # step in seconds
        voiced_segments = []
        in_voiced_region = False
        start_samp = 0

        for i, prob in enumerate(speech_probability):
            time_sec = frames.start + i * sr_per_frame
            sample_idx = int(time_sec * sr)
            if prob > threshold and not in_voiced_region:
                # start of speech region
                start_samp = sample_idx
                in_voiced_region = True
            elif prob <= threshold and in_voiced_region:
                # end of speech region
                end_samp = sample_idx
                in_voiced_region = False
                voiced_segments.append((start_samp, end_samp))

        # If we ended in a voiced region, close it
        if in_voiced_region:
            voiced_segments.append((start_samp, len(audio)))

        return voiced_segments

    def get_f0(
        self,
        input_audio_path: str,
        audio: np.ndarray,
        p_len: int,
        f0_up_key: float,
        f0_method: str,
        filter_radius: int,
        inp_f0: np.ndarray = None,
    ):
        """
        Extract F0 using one of several methods:
          - pm       : Parselmouth AC algorithm
          - harvest  : PyWorld's harvest + stonemask
          - crepe    : Crepe model
          - rmvpe    : Rmvpe model

        Also applies pitch shifting (f0_up_key).
        Optionally merges external pitch (inp_f0) for part of the audio.
        Returns (coarse_f0, continuous_f0).
        """
        time_step_ms = (self.window / self.sr) * 1000  # hop in ms
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        # ---------- Select f0 method ----------
        if f0_method == "pm":
            # Parselmouth
            f0 = (
                parselmouth.Sound(audio)
                .to_pitch_ac(
                    time_step=time_step_ms / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            # Pad or truncate to p_len
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or (p_len - len(f0) - pad_size) > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

        elif f0_method == "harvest":
            # Cache the audio globally
            input_audio_path2wav[input_audio_path] = audio.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            # Optional smoothing
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)

        elif f0_method == "crepe":
            # Crepe-based pitch extraction
            model_name = "full"
            batch_size = 512  # adjust for GPU memory
            audio_tensor = torch.tensor(np.copy(audio))[None].float()

            f0, periodicity = torchcrepe.predict(
                audio_tensor,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model_name,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            # Median filtering
            periodicity = torchcrepe.filter.median(periodicity, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[periodicity < 0.1] = 0
            f0 = f0[0].cpu().numpy()

        else:
            # RMVPE-based pitch extraction
            if self.model_rmvpe is None:
                from rvc.infer.lib.rmvpe import RMVPE
                rmvpe_model_path = os.path.join(model_path, "rvc", "rmvpe.pt")
                logger.info(f"Loading RMVPE model from {rmvpe_model_path}")
                self.model_rmvpe = RMVPE(
                    model_path=rmvpe_model_path,
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(audio, thred=0.03)

            # If running on MPS or XPU, free up memory (optional)
            if "privateuseone" in str(self.device):
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")

        # ---------- Apply pitch shift ----------
        f0 *= pow(2, f0_up_key / 12)

        # ---------- Merge external f0 (if provided) ----------
        tf0 = self.sr // self.window  # pitch frames per second
        if inp_f0 is not None:
            # Example: first column is time, second is freq
            delta_t = int(round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1))
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            start_idx = self.x_pad * tf0
            end_idx = start_idx + len(replace_f0)
            shape = f0[start_idx:end_idx].shape[0]
            f0[start_idx: start_idx + len(replace_f0)] = replace_f0[:shape]

        # ---------- Convert to coarse & continuous ----------
        f0_continuous = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700.0)
        # Scale into 1~255
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        return f0_coarse, f0_continuous

    def vc(
        self,
        model,
        net_g,
        sid,
        audio_chunk,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        """
        Convert a chunk of audio using the RVC pipeline.
        Handles stereo channels separately and combines them back.
        Then re-applies silence based on the original input's amplitude.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        from time import time as ttime

        def amplitude_db(x: float) -> float:
            """Utility to convert a sample's amplitude to decibels."""
            return 20.0 * np.log10(max(abs(x), 1e-8))

        def restore_silence(original_wave: np.ndarray, cloned_wave: np.ndarray, threshold_db=-60.0):
            """
            If the original wave amplitude is below threshold_db,
            force the cloned wave to be silent at that sample.
            """
            if original_wave.ndim == 1:
                # Mono
                num_samples = len(original_wave)
                for i in range(num_samples):
                    if amplitude_db(original_wave[i]) < threshold_db:
                        cloned_wave[i] = 0.0
            else:
                # Stereo (or multi-channel)
                num_samples, num_channels = original_wave.shape
                for i in range(num_samples):
                    for c in range(num_channels):
                        if amplitude_db(original_wave[i, c]) < threshold_db:
                            cloned_wave[i, c] = 0.0
            return cloned_wave

        # ---------- Convert input chunk to tensor ----------
        feats = torch.from_numpy(audio_chunk)
        feats = feats.half() if self.is_half else feats.float()

        # Separate stereo or mono
        if feats.dim() == 2:
            # Stereo
            feats_left = feats[:, 0].view(1, -1)
            feats_right = feats[:, 1].view(1, -1)
        else:
            # Mono
            feats_left = feats.view(1, -1)
            feats_right = None

        padding_mask_left = torch.BoolTensor(feats_left.shape).to(self.device).fill_(False)
        padding_mask_right = (
            torch.BoolTensor(feats_right.shape).to(self.device).fill_(False)
            if feats_right is not None
            else None
        )

        def process_channel(feats_channel, padding_mask_channel, pitch_vals, pitchf_vals):
            inputs = {
                "source": feats_channel.to(self.device),
                "padding_mask": padding_mask_channel,
                "output_layer": 9 if version == "v1" else 12,
            }
            t0 = ttime()
            with torch.no_grad():
                logits = model.extract_features(**inputs)
                feats_base = model.final_proj(logits[0]) if version == "v1" else logits[0]

            # ---------- Optional Faiss Index Search (k-NN) ----------
            if index is not None and big_npy is not None and index_rate != 0:
                feats_np = feats_base[0].cpu().numpy()
                feats_np = feats_np.astype("float32") if self.is_half else feats_np

                # top-8 neighbors
                score, ix = index.search(feats_np, k=8)
                weight = np.square(1.0 / (score + 1e-8))
                weight = weight / weight.sum(axis=1, keepdims=True)
                feats_neighbors = np.sum(big_npy[ix] * weight[:, :, None], axis=1)

                if self.is_half:
                    feats_neighbors = feats_neighbors.astype("float16")
                feats_neighbors_tensor = torch.from_numpy(feats_neighbors).unsqueeze(0).to(self.device)
                feats_base = feats_neighbors_tensor * index_rate + feats_base * (1 - index_rate)

            # ---------- Double time resolution ----------
            feats_base = F.interpolate(
                feats_base.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)

            p_len = feats_channel.shape[1] // self.window
            if feats_base.shape[1] < p_len:
                p_len = feats_base.shape[1]
                if pitch_vals is not None and pitchf_vals is not None:
                    pitch_vals = pitch_vals[:, :p_len]
                    pitchf_vals = pitchf_vals[:, :p_len]

            # ---------- Protective crossfade on unvoiced sections ----------
            if protect < 0.5 and (pitch_vals is not None and pitchf_vals is not None):
                feats_original = feats_base.clone()
                feats_base_length = feats_base.shape[1]
                feats_original = F.interpolate(
                    feats_original.permute(0, 2, 1),
                    size=feats_base_length,
                ).permute(0, 2, 1)
                pitch_mask = pitchf_vals.clone()
                pitch_mask[pitchf_vals > 0] = 1.0  # voiced
                pitch_mask[pitchf_vals < 1] = protect  # unvoiced
                pitch_mask = pitch_mask.unsqueeze(-1)
                feats_base = feats_base * pitch_mask + feats_original * (1 - pitch_mask)
                feats_base = feats_base.to(feats_original.dtype)

            p_len_tensor = torch.tensor([p_len], device=self.device).long()
            t1 = ttime()
            times[0] += (t1 - t0)  # feature extraction time

            with torch.no_grad():
                if pitch_vals is not None and pitchf_vals is not None:
                    audio_out = net_g.infer(
                        feats_base, p_len_tensor, pitch_vals, pitchf_vals, sid
                    )[0][0, 0]
                else:
                    audio_out = net_g.infer(feats_base, p_len_tensor, sid)[0][0, 0]

            t2 = ttime()
            times[2] += (t2 - t1)  # inference time

            return audio_out.data.cpu().float().numpy()

        # ---------- Process left & right ----------
        audio_out_left = process_channel(feats_left, padding_mask_left, pitch, pitchf)
        if feats_right is not None:
            audio_out_right = process_channel(feats_right, padding_mask_right, pitch, pitchf)
            audio_out = np.stack((audio_out_left, audio_out_right), axis=-1)
        else:
            audio_out = audio_out_left

        # ---------- Re-apply silence (avoid hiss in truly silent regions) ----------
        audio_out = restore_silence(original_wave=audio_chunk, cloned_wave=audio_out, threshold_db=-60.0)

        # Cleanup
        del feats_left, feats_right, padding_mask_left, padding_mask_right
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_out

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio: np.ndarray,
        input_audio_path: str,
        times: list,
        f0_up_key: float,
        f0_method: str,
        file_index: str,
        index_rate: float,
        if_f0: int,
        filter_radius: int,
        tgt_sr: int,
        resample_sr: int,
        rms_mix_rate: float,
        version: str,
        protect: float,
        f0_file=None,
    ) -> np.ndarray:
        """
        Main pipeline function that:
          1. Loads Faiss index if needed
          2. High-pass filters (48Hz)
          3. Extracts pitch if needed
          4. Uses Pyannote VAD to split into voiced segments
          5. For each segment, run vc() (skipping nearly silent ones)
          6. Crossfade adjacent segments
          7. Optionally mix RMS with original audio
          8. Resample if needed
          9. Convert to int16
          10. Returns final audio
        """

        import traceback
        import faiss
        from time import time as ttime

        # 1) Load Faiss index if needed
        index, big_npy = None, None
        if file_index and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception:
                traceback.print_exc()
                index = None
                big_npy = None

        # 2) High-pass filter below ~48Hz
        if self.bh is not None and self.ah is not None:
            audio = signal.filtfilt(self.bh, self.ah, audio)

        # 3) Prepare pitch if needed
        #    We pad the audio so that the length matches what RVC expects
        #    for stable pitch extraction.
        audio_pad2 = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad2.shape[0] // self.window
        inp_f0 = None

        if f0_file is not None and hasattr(f0_file, "name"):
            # Attempt to read external F0 data
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip().split("\n")
                parsed_f0 = []
                for line in lines:
                    parsed_f0.append([float(item) for item in line.split(",")])
                inp_f0 = np.array(parsed_f0, dtype="float32")
            except Exception:
                traceback.print_exc()

        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        pitch, pitchf = None, None
        t_start = ttime()
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad2,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            # Truncate if needed
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        t_pitch_ready = ttime()
        times[1] += (t_pitch_ready - t_start)

        # 4) Identify voiced segments with Pyannote
        #    We'll do it on the *un-padded* audio, sample rate is 16k
        vad_segments = self.voice_activity_segments(audio, sr=self.sr, threshold=0.6)
        if not vad_segments:
            # fallback: entire audio
            vad_segments = [(0, len(audio))]

        # We'll store the final output in float32, then convert
        # at the end. Initialize to zeros:
        output = np.zeros_like(audio, dtype=np.float32)

        # Crossfade length (in samples)
        crossfade = 512  # ~32ms @16k
        prev_end_samp = 0

        # 5) For each voiced segment, run vc() (skipping super-silent ones)
        for (start_samp, end_samp) in vad_segments:
            # If region is extremely quiet, skip
            region_db = self.amplitude_db(audio[start_samp:end_samp])
            if region_db < -55:
                # Just keep it silent
                continue

            # Add some padding (t_pad) around the segment
            pad_start = max(0, start_samp - self.t_pad)
            pad_end = min(len(audio), end_samp + self.t_pad)
            # Extract chunk
            chunk = audio[pad_start:pad_end].copy()

            # Now figure out pitch slices for this chunk if needed
            if if_f0 == 1:
                # Convert from sample indexes to pitch indexes
                # Each pitch frame is self.window samples
                pitch_start = pad_start // self.window
                pitch_end = pad_end // self.window
                pitch_vals = pitch[:, pitch_start:pitch_end] if pitch is not None else None
                pitchf_vals = pitchf[:, pitch_start:pitch_end] if pitchf is not None else None
            else:
                pitch_vals, pitchf_vals = None, None

            # Run VC on the chunk
            converted_chunk = self.vc(
                model=model,
                net_g=net_g,
                sid=sid_tensor,
                audio_chunk=chunk,
                pitch=pitch_vals,
                pitchf=pitchf_vals,
                times=times,
                index=index,
                big_npy=big_npy,
                index_rate=index_rate,
                version=version,
                protect=protect,
            )

            # Place it back into the output with crossfade around [start_samp, end_samp]
            # Figure out local offsets
            fade_start = start_samp - pad_start
            fade_end = fade_start + (end_samp - start_samp)

            # 6) Crossfade with the existing output
            #    Crossfade region: [start_samp - crossfade, start_samp]
            fade_len = min(crossfade, fade_start)  # can't exceed chunk boundary
            for i in range(fade_len):
                alpha = i / float(fade_len)
                out_idx = (start_samp - fade_len + i)
                chunk_idx = (fade_start - fade_len + i)
                if 0 <= out_idx < len(output) and 0 <= chunk_idx < len(converted_chunk):
                    old_val = output[out_idx]
                    new_val = converted_chunk[chunk_idx]
                    output[out_idx] = (1 - alpha) * old_val + alpha * new_val

            # Copy the main portion (non-crossfade)
            main_start = start_samp
            main_end = end_samp
            chunk_main_start = fade_start
            chunk_main_end = fade_start + (end_samp - start_samp)
            chunk_main_end = min(chunk_main_end, len(converted_chunk))

            out_len = main_end - main_start
            for i in range(out_len):
                out_idx = main_start + i
                chunk_idx = chunk_main_start + i
                if out_idx < len(output) and chunk_idx < len(converted_chunk):
                    output[out_idx] = converted_chunk[chunk_idx]

            prev_end_samp = end_samp

        # 7) Optionally mix RMS with original audio
        if rms_mix_rate != 1.0:
            output = change_rms(audio, self.sr, output, self.sr, rms_mix_rate)

        # 8) Resample if needed
        if tgt_sr != resample_sr and resample_sr >= 16000:
            output = librosa.resample(output, orig_sr=tgt_sr, target_sr=resample_sr)

        # 9) Convert to int16
        audio_max = np.abs(output).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        output_int16 = (output * max_int16).astype(np.int16)

        # Cleanup
        del pitch, pitchf, sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 10) Return final audio
        return output_int16
