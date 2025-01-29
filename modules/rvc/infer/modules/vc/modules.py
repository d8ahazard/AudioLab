import logging
import os
import traceback
import uuid

import librosa
import numpy as np
import soundfile as sf
import torch

from handlers.config import model_path
from handlers.reverb import apply_reverb
from handlers.stereo import stereo_to_mono_ms, resample_side, mono_to_stereo_ms
from modules.rvc.infer.lib.audio import load_audio
from modules.rvc.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from modules.rvc.infer.modules.vc.pipeline import Pipeline
from modules.rvc.infer.modules.vc.utils import get_index_path_from_model, load_hubert

logger = logging.getLogger(__name__)


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

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
                self.version = self.cpt.get("version", "v2")
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
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect1
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"},
            "", ""
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
            resample_sr,
            rms_mix_rate,
            protect,
            reverb_param_path=None,
            do_noise_removal=False,
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
            audio_float, og_sr = load_audio(
                file=input_audio_path,
                sr=None,  # Do NOT resample, keep original
                mono=False,  # Keep stereo if applicable
                return_sr=True
            )

            logger.info(f"Loaded audio: shape={audio_float.shape}, original sample rate={og_sr}")

            # Normalize if necessary
            audio_float /= max(1.0, np.abs(audio_float).max() / 0.95)

            # ----------------------------------------------------------------
            # (B) Handle Stereo -> Mono conversion (Mid-Side Encoding)
            # ----------------------------------------------------------------
            if audio_float.ndim == 2 and audio_float.shape[1] == 2:
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
            file_index = model.replace("_final.pth", ".index")
            file_index = os.path.join(model_path, "trained", file_index)

            if self.pipeline is None:
                self.get_vc(model)

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            logger.info("Running pipeline...")
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
            if do_noise_removal:
                logger.info("Applying noise removal...")
                from handlers.noise_removal import restore_silence
                if final_float.shape[1] == 1:
                    final_float = restore_silence(mono_og, final_float[:, 0]).reshape(-1, 1)

            # ----------------------------------------------------------------
            # (H) Apply Reverb (if enabled)
            # ----------------------------------------------------------------
            if reverb_param_path:
                logger.info("Applying reverb...")
                temp_input = f"temp_in_{uuid.uuid4()}.wav"
                temp_output = f"temp_out_{uuid.uuid4()}.wav"

                sf.write(temp_input, final_float, og_sr, format="WAV", subtype="PCM_16")
                apply_reverb(temp_input, reverb_param_path, temp_output)
                wet_signal, _ = load_audio(temp_output, sr=og_sr, mono=False)
                os.remove(temp_input)
                os.remove(temp_output)
                final_float = wet_signal

            # ----------------------------------------------------------------
            # (I) Final Amplitude Scaling & Clipping
            # ----------------------------------------------------------------
            max_val = np.max(np.abs(final_float))
            if max_val > 1.0:
                logger.info(f"Clipping audio with max value: {max_val}")
                final_float /= max_val  # Simple limiting
            final_float = np.clip(final_float, -1.0, 1.0)

            return f"Success. Processing time: {times}", (og_sr, final_float)

        except Exception as e:
            info = traceback.format_exc()
            logger.info("[vc_single ERROR]", info)
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
            resample_sr,
            rms_mix_rate,
            protect,
            format1,
            project_dir,
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
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        # Extract base name and append "(Cloned)"
                        base_name, ext = os.path.splitext(os.path.basename(path))
                        cloned_name = f"{base_name}(Cloned).wav"

                        output_file = os.path.join(opt_root, f"{cloned_name}")

                        try:
                            # Save the processed audio
                            sf.write(output_file, audio_opt, tgt_sr, format="wav", subtype="PCM_16")
                            outputs.append(output_file)
                        except Exception as e:
                            logger.info(f"Error saving audio file: {e}")
                            traceback.print_exc()

                    except Exception as e:
                        traceback.print_exc()

        except:
            traceback.print_exc()
            logger.warning(traceback.format_exc())
        return outputs
