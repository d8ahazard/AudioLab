import logging
import time
import traceback
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from rvc.infer.lib.rmvpe import RMVPE

logger = logging.getLogger(__name__)

# Constants for high-pass filter
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype='high', fs=SAMPLE_RATE)


class AudioProcessor:
    @staticmethod
    def change_rms(source_audio: np.ndarray, source_rate: int, target_audio: np.ndarray, target_rate: int,
                   rate: float) -> np.ndarray:
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.
        """
        # Calculate RMS of both audio data
        rms1 = librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)
        rms2 = librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)

        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(torch.from_numpy(rms1).float().unsqueeze(0), size=target_audio.shape[0],
                             mode='linear').squeeze()
        rms2 = F.interpolate(torch.from_numpy(rms2).float().unsqueeze(0), size=target_audio.shape[0],
                             mode='linear').squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        return adjusted_audio


class Pipeline:
    def __init__(self, rmvpe_path: str, config):
        # Initializing various configurations from the provided config object
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.is_half = config.is_half
        self.device = config.device

        # Don't forget to set up t_pad_tgt with set_t_pad_tgt() before using the pipeline
        self.t_pad_tgt = None

        self.sr = 16000  # Sample rate for input audio
        self.window = 160  # Frame size
        self.t_pad = self.sr * self.x_pad  # Padding duration
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # Query duration
        self.t_center = self.sr * self.x_center  # Center position for querying
        self.t_max = self.sr * self.x_max  # Max duration for querying
        # Initialize RMVPE model
        self.rmvpe_model = RMVPE(model_path=str(rmvpe_path), is_half=self.is_half, device=self.device)

    def set_t_pad_tgt(self, tgt_sr: int):
        """
        Sets the target padding duration.
        This value is used to trim the output audio.
        """
        self.t_pad_tgt = tgt_sr * self.x_pad

    def get_mel(self, audio: np.ndarray) -> np.ndarray:
        mel = self.rmvpe_model.mel_extractor(
            torch.from_numpy(audio).float().to(self.device).unsqueeze(0), center=True
        )
        return mel.squeeze().cpu().numpy()

    def get_f0(self, x: np.ndarray, f0_up_key: Union[int, str],
               inp_f0: Optional[np.ndarray] = None, average_pitch: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts F0 contour from the given audio signal `x` using the RMVPE method.
        `f0_up_key` is used for pitch shifting. Can be 'auto' then pitch shift will be derived from difference in f0
        `inp_f0` (if provided) is used to replace the extracted F0 contour with custom values.
        """
        # Extract the F0 contour
        f0 = self.rmvpe_model.infer_from_audio(x, thred=0.03)

        if f0_up_key == 'auto':
            if average_pitch > 0:
                pitch_input = f0[f0 > 25]
                lower_bound = np.percentile(pitch_input, 5)
                upper_bound = np.percentile(pitch_input, 95)
                src_average_pitch = np.mean(pitch_input[(pitch_input > lower_bound) & (pitch_input < upper_bound)])
                # f0_up_key should be in range -12 to 12, clipped, derived from the difference src target
                f0_up_key = int(np.clip(np.round(12 * np.log2(average_pitch / src_average_pitch)), -12, 12))
                print("pitch shift auto set to", f0_up_key)
            else:
                f0_up_key = 0

        # Pitch shifting
        f0 *= pow(2, f0_up_key / 12)

        # Replace the F0 contour if custom F0 (`inp_f0`) is provided
        if inp_f0 is not None:
            tf0 = self.sr // self.window  # F0 points per second
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        # Convert F0 to mel scale and quantize
        f0_mel_min, f0_mel_max = 1127 * np.log1p([f0.min() / 700, f0.max() / 700])
        f0_mel = 1127 * np.log1p(f0 / 700)
        f0_mel = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255)
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        return f0_coarse, f0

    def apply_index(self, feats, index, big_npy, index_rate, k=8):
        """
        Apply the index to the given features.
        After retrieving top-k features from the index, the features are blended with the original features.

        The higher the `index_rate`, the more the feature from train dataset are blended with the original features.
        """
        if (
                not isinstance(index, type(None))
                and not isinstance(big_npy, type(None))
                and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            # TAKE top-8 features and do weighted average
            score, ix = index.search(npy, k=k)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            logger.info(f"top-{k} weights from index (after renorm): \n{weight}")
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                    + (1 - index_rate) * feats
            )
        return feats

    def vc(self,
           feature_extractor,
           net_g,
           sid,
           audio0,
           pitch,
           pitchf,
           times,
           index,
           big_npy,
           index_rate,
           protect,
           index_topk: int = 8):
        """Voice conversion function to modify and convert the speaker's identity.

        Args:
            feature_extractor: The loaded HuBERT model for feature extraction.
            net_g: The generator network for synthesizing audio.
            sid: Speaker ID for target speaker's voice characteristics.
            audio0: The input audio features.
            pitch: F0 contour (pitch information) of the input audio.
            pitchf: Full-band pitch contour of the input audio.
            times: List for keeping track of processing times.
            index: FAISS index for nearest neighbor search in the embedding space.
            big_npy: The embeddings corresponding to the FAISS index.
            index_rate: The blending ratio for mixing the original and matched embeddings.
            protect: Ratio for protecting the prosody during conversion.
            index_topk: The number of top embeddings to retrieve from the index.

        Returns:
            The converted audio as a NumPy array.
        """
        has_pitch = pitch is not None and pitchf is not None

        # HUBERT PART
        feats = torch.from_numpy(audio0).unsqueeze(0)
        feats = feats.half() if self.is_half else feats.float()
        assert feats.dim() == 2, f"Expected 2D tensor, but got {feats.dim()}D tensor"
        feats = feats.mean(dim=-1) if feats.shape[1] == 2 else feats

        # Preparing input for feature extraction
        padding_mask = torch.full(feats.shape, False, dtype=torch.bool, device=self.device)
        hubert_inputs = {"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 12}

        t0 = time.time()
        # Feature extraction using the model
        with torch.no_grad(), torch.autocast("cuda", enabled=self.is_half):
            logger.info("Extracting features from HuBERT")
            logits = feature_extractor.extract_features(**hubert_inputs)
            feats = logits[0]  # Since we are using v2 only, we do not use final_proj

        # INDEX PART
        if protect < 0.5 and has_pitch:
            feats0 = feats.clone()
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )

        try:
            feats = self.apply_index(feats, index, big_npy, index_rate, k=index_topk)
        except:
            logger.info("Apply index failed, using original features")
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        t1 = time.time()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and has_pitch:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad(), torch.autocast("cuda", enabled=self.is_half):
            arg = (feats, p_len, pitch, pitchf, sid) if has_pitch else (feats, p_len, sid)

            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            logger.info(f"audio1 shape {audio1.shape}")
            del has_pitch, arg
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = time.time()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
            self,
            feature_extractor,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            file_index,
            index_rate,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            protect,
            f0_file=None,
            index_topk: int = 8,
            average_pitch: int = 0
    ):
        """
        f0_up_key: Shift predicted F0 contour by this many semitones. (Default: 0) Acceptable range: -12 to 12.
        tgt_sr: Target sample rate for the converted audio.
        resample_sr: Sample rate for resampling operations.
        rms_mix_rate: Rate for RMS-based mixing.
        protect: Ratio for protecting the prosody during conversion.
        f0_file: Optional file containing F0 contour information.

        """
        index = faiss.read_index(file_index)
        # big_npy = np.load(file_big_npy)
        big_npy = index.reconstruct_n(0, index.ntotal)

        audio = signal.filtfilt(bh, ah, audio)
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
        t = None
        t1 = time.time()
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

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = self.get_f0(
            audio_pad,
            f0_up_key,
            inp_f0,
            average_pitch=average_pitch
        )
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]
        if "mps" not in str(self.device) or "xpu" not in str(self.device):
            pitchf = pitchf.astype(np.float32)
        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        logger.info(f"got pitch {pitch.shape}")

        # compute mean of medians of f0 from each window for analysis
        f0_statistics = []

        t2 = time.time()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                self.vc(
                    feature_extractor,
                    net_g,
                    sid,
                    audio_pad[s: t + self.t_pad2 + self.window],
                    pitch[:, s // self.window: (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window: (t + self.t_pad2) // self.window],
                    times,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                    index_topk=index_topk,
                )[self.t_pad_tgt: -self.t_pad_tgt]
            )
            s = t
        audio_opt.append(
            self.vc(
                feature_extractor,
                net_g,
                sid,
                audio_pad[t:],
                pitch[:, t // self.window:] if t is not None else pitch,
                pitchf[:, t // self.window:] if t is not None else pitchf,
                times,
                index,
                big_npy,
                index_rate,
                protect,
                index_topk=index_topk,
            )[self.t_pad_tgt: -self.t_pad_tgt]
        )

        if t is not None:
            pitch_input = pitchf[:, t // self.window:].cpu().numpy()
        else:
            pitch_input = pitchf.cpu().numpy()
        pitch_input = pitch_input[pitch_input > 25]
        lower_bound = np.percentile(pitch_input, 5)
        upper_bound = np.percentile(pitch_input, 95)
        f0_statistics.append(lower_bound)
        f0_statistics.append(upper_bound)
        f0_statistics.append(np.mean(pitch_input[(pitch_input > lower_bound) & (pitch_input < upper_bound)]))
        logger.info(f"f0_statistics: {f0_statistics}")

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = AudioProcessor.change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)

        if tgt_sr != resample_sr and resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"return len {len(audio_opt)}")
        return audio_opt, f0_statistics
