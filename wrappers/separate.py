import hashlib
import json
import os
import re
import threading
from typing import Any, List, Dict

import librosa
import numpy as np
import soundfile as sf

from handlers.config import model_path, output_path
from handlers.reverb import extract_reverb
from modules.audio_separator.audio_separator import separate_music
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput
import logging

logger = logging.getLogger(__name__)


class Separate(BaseWrapper):
    """
    An advanced wrapper that:
      1) Separates the audio into multiple stems or just vocals,
      2) Optionally removes reverb/echo/noise/crowd,
      3) Can do background-vocal splitting,
      4) Can do advanced drum separation if requested.
    """

    def register_api_endpoint(self, api) -> Any:
        pass

    title = "Separate"
    priority = 1
    separator = None
    default = True
    required = True
    description = (
        "Separate audio into distinct stems (vocals, instruments, bass, drums, etc.), "
        "optionally remove reverb, echo/delay, crowd noise, and general background noise."
    )
    file_operation_lock = threading.Lock()

    allowed_kwargs = {
        "delete_extra_stems": TypedInput(
            default=True,
            description="Automatically delete intermediate stem files after processing to save disk space. Disable this if you want to retain all intermediate outputs.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_bg_vocals": TypedInput(
            default=True,
            description="Separates background vocals from the main vocals, creating an additional stem for detailed manipulation.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_stems": TypedInput(
            default=False,
            description="Enable to separate the audio into distinct stems, such as vocals, instruments, and percussion. Useful for remixing or isolating specific elements.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "store_reverb_ir": TypedInput(
            default=True,
            description="Store the impulse response for reverb removal. This allows for reapplying the reverb later during stem merging. Note: This is an experimental feature and may not always produce accurate results.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_drums": TypedInput(
            default=False,
            description="Separate the drum track from the rest of the audio. This is useful for remixing or adjusting the drum track independently. Requires 'Separate Stems' to be enabled.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "separate_woodwinds": TypedInput(
            default=False,
            description="Separate the woodwind instruments from the rest of the audio. This is useful for remixing or adjusting the woodwind track independently. Requires 'Separate Stems' to be enabled.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "alt_bass_model": TypedInput(
            default=False,
            description="Use an alternative bass model for better bass separation. This may improve the quality of the bass stem if the bass part was played on an electric/plucked bass.",
            type=bool,
            gradio_type="Checkbox"
        ),
        # Large-scale removal toggles
        "reverb_removal": TypedInput(
            default="Main Vocals",
            description="Choose the scope of reverb removal. Options include leaving it unchanged, applying it to main vocals only, all vocals, or all stems.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "echo_removal": TypedInput(
            default="Nothing",
            description="Specify the level of echo removal. You can remove echo from nothing, main vocals, all vocals, or all stems.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "delay_removal": TypedInput(
            default="Nothing",
            description="Specify the target for delay removal: nothing, main vocals, all vocals, or all stems. This reduces delay artifacts in the audio.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "crowd_removal": TypedInput(
            default="Nothing",
            description="Remove crowd noise from selected stems. Options include removing it from nothing, main vocals, all vocals, or all stems.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "noise_removal": TypedInput(
            default="Nothing",
            description="Remove general background noise from the selected stems. Choose from nothing, main vocals, all vocals, or all stems.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        # Model picks (optional if you want to override defaults)
        "delay_removal_model": TypedInput(
            default="UVR-DeEcho-DeReverb.pth",
            description="Which echo/delay removal model to use",
            type=str,
            choices=["UVR-DeEcho-DeReverb.pth", "UVR-De-Echo-Normal.pth"],
            gradio_type="Dropdown"
        ),
        "noise_removal_model": TypedInput(
            default="UVR-DeNoise.pth",
            description="Choose the model used for noise removal. The 'Lite' version may perform faster but with slightly reduced quality.",
            type=str,
            choices=["UVR-DeNoise.pth", "UVR-DeNoise-Lite.pth"],
            gradio_type="Dropdown"
        ),
        "crowd_removal_model": TypedInput(
            default="UVR-MDX-NET_Crowd_HQ_1.onnx",
            description="Select the model for removing crowd noise. Different models may excel in specific environments or audio scenarios.",
            type=str,
            choices=["UVR-MDX-NET_Crowd_HQ_1.onnx", "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"],
            gradio_type="Dropdown"
        ),
    }

    def setup(self):
        """
        Ensure the main models are downloaded using the same library (audio_separator.separator).
        We might also pre-download reverb/noise/echo removal models here if we want to chain them.
        """
        from audio_separator.separator import Separator

        model_dir = os.path.join(model_path, "audio_separator")
        os.makedirs(model_dir, exist_ok=True)
        out_dir = os.path.join(output_path, "audio_separator")
        os.makedirs(out_dir, exist_ok=True)

        self.separator = Separator(log_level=logging.ERROR, model_file_dir=model_dir, output_dir=out_dir)

        needed = [
            "deverb_bs_roformer_8_384dim_10depth.ckpt",
            "UVR-DeEcho-DeReverb.pth",
            "UVR-De-Echo-Normal.pth",
            "UVR-DeNoise.pth",
            "UVR-DeNoise-Lite.pth",
            "UVR-MDX-NET_Crowd_HQ_1.onnx",
            "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
        ]
        for m in needed:
            self.separator.download_model_files(m)

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        """
        The big method:
          1) Potentially skip separation if the stems/ config already matches,
          2) If needed, calls `separate_music` to get stems,
          3) Optionally splits background vocals from main,
          4) Then runs reverb/echo/delay/crowd/noise removal in the user-specified way,
          5) Saves a JSON for caching separation results,
          6) Returns final stems.
        """
        self.setup()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        # Basic toggles
        separate_stems = filtered_kwargs.get("separate_stems", False)
        separate_drums = filtered_kwargs.get("separate_drums", False)
        separate_woodwinds = filtered_kwargs.get("separate_woodwinds", False)
        alt_bass_model = filtered_kwargs.get("alt_bass_model", False)
        separate_bg_vocals = filtered_kwargs.get("separate_bg_vocals", True)
        delete_extra_stems = filtered_kwargs.get("delete_extra_stems", True)
        store_reverb_ir = filtered_kwargs.get("store_reverb_ir", False)

        # Removal toggles
        reverb_removal = filtered_kwargs.get("reverb_removal", "Nothing")
        echo_removal = filtered_kwargs.get("echo_removal", "Nothing")
        delay_removal = filtered_kwargs.get("delay_removal", "Nothing")
        crowd_removal = filtered_kwargs.get("crowd_removal", "Nothing")
        noise_removal = filtered_kwargs.get("noise_removal", "Nothing")

        # Model picks
        delay_removal_model = filtered_kwargs.get("delay_removal_model", "UVR-DeEcho-DeReverb.pth")
        noise_removal_model = filtered_kwargs.get("noise_removal_model", "UVR-DeNoise.pth")
        crowd_removal_model = filtered_kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")

        all_generated = []
        outputs = []
        projects_out = []

        for project in inputs:
            proj_stems = []
            input_file = project.src_file
            out_dir = os.path.join(project.project_dir, "stems")
            self.separator.output_dir = out_dir
            os.makedirs(out_dir, exist_ok=True)

            # 1) Check if there's a JSON with existing separation info
            cache_file = os.path.join(out_dir, "separation_info.json")
            skip_separation = False
            if os.path.exists(cache_file):
                # Read the cache
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)

                # Compare config
                # Make a small dict of current separation config
                current_config = {
                    "file": input_file,
                    "separate_stems": separate_stems,
                    "separate_drums": separate_drums,
                    "separate_woodwinds": separate_woodwinds,
                    "alt_bass_model": alt_bass_model,
                    "separate_bg_vocals": separate_bg_vocals,
                    # removal toggles
                    "reverb_removal": reverb_removal,
                    "echo_removal": echo_removal,
                    "delay_removal": delay_removal,
                    "crowd_removal": crowd_removal,
                    "noise_removal": noise_removal,
                    # model picks
                    "delay_removal_model": delay_removal_model,
                    "noise_removal_model": noise_removal_model,
                    "crowd_removal_model": crowd_removal_model,
                    "store_reverb_ir": store_reverb_ir,
                }

                # If the configs match exactly, check files & hashes
                if cached_data.get("config") == current_config:
                    # verify all stems exist and match
                    output_stems = []
                    all_stems_good = True
                    for stem_info in cached_data.get("stems", []):
                        path = stem_info["path"]
                        output_stems.append(path)
                        digest = stem_info["hash"]

                        if not os.path.exists(path):
                            all_stems_good = False
                            break
                        if self._hash_file(path) != digest:
                            all_stems_good = False
                            break

                    if all_stems_good:
                        # We can skip separation; retrieve the final stems from JSON
                        # Our code also manipulates stems after separation, so let's gather them:
                        final_stem_paths = [st["path"] for st in cached_data["stems"]]
                        proj_stems.extend(final_stem_paths)
                        outputs.extend(final_stem_paths)
                        project.add_output("stems", final_stem_paths)
                        projects_out.append(project)
                        continue

            # 2) If we don't skip separation, do the separation process
            if not skip_separation:
                # 2A) Actual separation
                separated_stems = separate_music(
                    input_audio=[input_file],
                    output_folder=out_dir,
                    cpu=False,
                    overlap_demucs=0.1,
                    overlap_VOCFT=0.1,
                    overlap_VitLarge=1,
                    overlap_InstVoc=1,
                    weight_InstVoc=8,
                    weight_VOCFT=1,
                    weight_VitLarge=5,
                    single_onnx=False,
                    large_gpu=True,
                    BigShifts=7,
                    vocals_only=not separate_stems,
                    use_VOCFT=True,
                    output_format="FLOAT",
                    callback=callback,
                    separate_drums=separate_drums,
                    separate_woodwinds=separate_woodwinds,
                    alt_bass_model=alt_bass_model
                )
                all_generated.extend(separated_stems)

                # 2B) Optionally do BG Vocal splitting
                updated_stems = []
                for full_path in separated_stems:
                    base_name = os.path.basename(full_path)
                    if "(Vocals)" in base_name and separate_bg_vocals:
                        self.separator.load_model("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
                        out_files = self.separator.separate(full_path)
                        out_files_full = [os.path.join(out_dir, p) for p in out_files]
                        out_files = [self._rename_bgvocal(p) for p in out_files_full]
                        all_generated.extend(out_files)
                        updated_stems.extend(out_files)  # [BG_Vocals, Vocals]
                    else:
                        updated_stems.append(full_path)

                # 2C) Transform chain: reverb -> echo -> delay -> crowd -> noise
                transformations = [
                    ("deverb_bs_roformer_8_384dim_10depth.ckpt", "No Reverb", reverb_removal),
                    (delay_removal_model, "No Echo", echo_removal),
                    (delay_removal_model, "No Delay", delay_removal),
                    (crowd_removal_model, "No Crowd", crowd_removal),
                    (noise_removal_model, "No Noise", noise_removal),
                ]

                final_stem_paths = []
                for stem_path in updated_stems:
                    current_path = stem_path
                    for model_file, out_label, transform_flag in transformations:
                        if self._should_apply_transform(os.path.basename(current_path), transform_flag):
                            self.separator.load_model(model_file)
                            partial = self.separator.separate(current_path)
                            partial_full = [os.path.join(out_dir, p) for p in partial]
                            all_generated.extend(partial_full)

                            pick, alt = None, None
                            if len(partial_full) == 2:
                                if out_label.replace(" ", "") in partial_full[0]:
                                    pick = partial_full[0]
                                    alt = partial_full[1]
                                else:
                                    pick = partial_full[1]
                                    alt = partial_full[0]

                                current_path = self._rename_file(os.path.basename(input_file), pick)

                                # If reverb removal from main vocals, maybe store IR
                                if out_label == "No Reverb" and "(Main Vocals)" in stem_path and store_reverb_ir and alt:
                                    try:
                                        out_ir = os.path.join(project.project_dir, "impulse_response.ir")
                                        logger.info(f"Extracting reverb IR from {os.path.basename(alt)}")
                                        extract_reverb(current_path, alt, out_ir)
                                    except Exception as e:
                                        print("Error extracting IR:", e)
                            else:
                                # if we have more or fewer than 2 stems
                                for pf in partial_full:
                                    if out_label.replace(" ", "") in pf:
                                        current_path = self._rename_file(os.path.basename(input_file), pf)
                                        break
                    final_stem_paths.append(current_path)

                proj_stems.extend(final_stem_paths)
                project.add_output("stems", proj_stems)
                outputs.extend(final_stem_paths)
                projects_out.append(project)

                # 3) Write out a JSON to skip future repeated separation
                cache_info = {
                    "config": {
                        "file": input_file,
                        "separate_stems": separate_stems,
                        "separate_drums": separate_drums,
                        "separate_woodwinds": separate_woodwinds,
                        "alt_bass_model": alt_bass_model,
                        "separate_bg_vocals": separate_bg_vocals,
                        "reverb_removal": reverb_removal,
                        "echo_removal": echo_removal,
                        "delay_removal": delay_removal,
                        "crowd_removal": crowd_removal,
                        "noise_removal": noise_removal,
                        "delay_removal_model": delay_removal_model,
                        "noise_removal_model": noise_removal_model,
                        "crowd_removal_model": crowd_removal_model,
                        "store_reverb_ir": store_reverb_ir
                    },
                    "stems": []
                }

                # Include paths + file hashes
                for p in final_stem_paths:
                    hash_val = self._hash_file(p)
                    cache_info["stems"].append({
                        "path": p,
                        "hash": hash_val,
                    })

                with open(cache_file, "w") as f:
                    json.dump(cache_info, f, indent=2)
            # End if not skip_separation

        # 4) Clean up intermediate if requested
        if delete_extra_stems:
            for p in all_generated:
                if p not in outputs and os.path.exists(p):
                    self.del_stem(p)

        return projects_out

    def test_vocal_models(self, input_file: str):
        """
        Test the vocal separation models by running them on a single input track.
        Layer vocal and instrumental/other outputs into combined tracks using weighted contributions.
        Normalize and apply noise removal based on averaged min/max amplitudes.
        """
        self.setup()
        out_dir = os.path.join(output_path, "test_vocal_models")
        os.makedirs(out_dir, exist_ok=True)

        # Define models with weights for vocals and instrumentals
        bg_model = [
            ("model_bs_roformer_ep_368_sdr_12.9628.ckpt", 8.4, 16.0),  # vocals (8.4), instrumental (16.0)
            ("MDX23C-8KFFT-InstVoc_HQ_2.ckpt", 7.2, 14.9),  # vocals (7.2), instrumental (14.9)
            ("UVR-MDX-NET-Voc_FT.onnx", 6.9, 14.9),  # vocals (6.9), instrumental (14.9)
            ("Kim_Vocal_2.onnx", 6.9, 14.9),  # vocals (6.9), instrumental (14.9)
            ("Kim_Vocal_1.onnx", 6.8, 14.9),  # vocals (6.8), instrumental (14.9)
            ("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", 6.8, 16.4),
            # vocals (6.8), instrumental (16.4)
        ]

        all_vocals = []
        all_instrumentals = []
        vocal_weights = []
        instrumental_weights = []
        sr = None  # Sampling rate
        max_amplitude = 0  # Track max amplitude across all files

        self.separator.output_dir = out_dir
        input_base, _ = os.path.splitext(os.path.basename(input_file))
        for background_vocal_model, vocal_weight, instrumental_weight in bg_model:
            model_base = background_vocal_model.replace('.', '_')  # Handle periods in filenames
            vocal_file = os.path.join(out_dir, f"{input_base}_(Vocals)_{model_base}.wav")
            instrumental_file = os.path.join(out_dir, f"{input_base}_(Instrumental)_{model_base}.wav")

            if not os.path.exists(vocal_file) or not os.path.exists(instrumental_file):
                self.separator.load_model(background_vocal_model)
                files = self.separator.separate(input_file)
                files = [os.path.join(out_dir, f) for f in files]

                vocal_file = files[0] if "(Vocals)" in files[0] else files[1]
                instrumental_file = files[1] if "(Vocals)" in files[0] else files[0]

            # Load files and track max amplitude
            vocals, sr = librosa.load(vocal_file, sr=None, mono=False)
            instrumentals, _ = librosa.load(instrumental_file, sr=None, mono=False)

            max_amplitude = max(
                max_amplitude,
                np.max(np.abs(vocals)),
                np.max(np.abs(instrumentals)),
            )

            # Weight and store outputs
            all_vocals.append(vocals * vocal_weight)
            all_instrumentals.append(instrumentals * instrumental_weight)
            vocal_weights.append(vocal_weight)
            instrumental_weights.append(instrumental_weight)

        # Combine and normalize outputs
        def combine_and_normalize(tracks, weights):
            max_length = max(track.shape[-1] for track in tracks)
            combined = np.zeros((tracks[0].shape[0], max_length), dtype=np.float32)  # Stereo

            for i, track in enumerate(tracks):
                combined[:, :track.shape[-1]] += track / sum(weights)

            # Normalize to max amplitude
            combined /= np.max(np.abs(combined))
            combined *= max_amplitude
            return combined

        combined_vocals = combine_and_normalize(all_vocals, vocal_weights)
        combined_instrumentals = combine_and_normalize(all_instrumentals, instrumental_weights)

        # Noise removal
        def remove_noise(audio):
            # Compute average of min/max amplitudes for noise threshold
            non_silent = audio[np.abs(audio) > 0.01]  # Avoid zeros
            if non_silent.size > 0:
                threshold = (np.min(non_silent) + np.max(non_silent)) / 2
                audio[np.abs(audio) < threshold] = 0  # Zero out low-amplitude noise
            return audio

        combined_vocals = remove_noise(combined_vocals)
        combined_instrumentals = remove_noise(combined_instrumentals)

        # Save final files
        vocal_file = os.path.join(out_dir, f"{input_base}_Combined_Vocals.wav")
        instrumental_file = os.path.join(out_dir, f"{input_base}_Combined_Instrumentals.wav")
        sf.write(vocal_file, combined_vocals.T, sr)  # Transpose for stereo
        sf.write(instrumental_file, combined_instrumentals.T, sr)  # Transpose for stereo

        return vocal_file, instrumental_file

    def del_stem(self, path: str) -> bool:
        try:
            with self.file_operation_lock:
                if os.path.exists(path):
                    os.remove(path)
                    return True
        except Exception as e:
            print(f"Error deleting {path}: {e}")
        return False

    @staticmethod
    def _should_apply_transform(stem_name: str, setting: str) -> bool:
        """
        Returns True if we should apply a transform to this stem, based on user setting
        (Nothing, All, All Vocals, Main Vocals).
        """
        if setting == "Nothing":
            return False
        if setting == "All":
            return True
        if setting == "All Vocals":
            return "Vocals)" in stem_name
        if setting == "Main Vocals":
            return "Vocals)" in stem_name and "(BG_Vocals)" not in stem_name
        return False

    @staticmethod
    def _rename_bgvocal(filepath: str) -> str:
        """
        For the Karaoke sub-separation of main vocals, the 'instrumental' track
        is actually BG vocals.
        Example: 'SongName_(Instrumental).wav' => 'SongName_(BG_Vocals).wav'
        """
        dirname = os.path.dirname(filepath)
        oldbase = os.path.basename(filepath)
        newbase = oldbase.replace("(Vocals)_(Instrumental)", "(Vocals)(BG_Vocals)")
        newbase = newbase.replace("(Vocals)_(Vocals)", "(Vocals)(Main Vocals)")
        newbase = newbase.replace("(Vocals)(Instrumental)", "(Vocals)(BG_Vocals)")
        newbase = newbase.replace("(Vocals)(Vocals)", "(Vocals)(Main Vocals)")
        newbase = newbase.replace("_mel_band_roformer_karaoke_aufr33_viperx_sdr_10", "")
        newpath = os.path.join(dirname, newbase)
        if os.path.exists(filepath):
            if os.path.exists(newpath):
                os.remove(newpath)
            os.rename(filepath, newpath)
        return newpath

    @staticmethod
    def _rename_file(base_in: str, filepath: str) -> str:
        """
        Rebuild the filename to remove model references, keep only user-friendly tags.
        Example: <original_song>_(Vocals)(No Reverb).wav
        """
        dirname = os.path.dirname(filepath)
        ext = os.path.splitext(filepath)[1]
        base_only = os.path.splitext(os.path.basename(base_in))[0]
        all_parens = re.findall(r"\([^)]*\)", os.path.basename(filepath))

        to_strip = [
            "deverb_bs_roformer", "UVR-DeEcho-DeReverb", "UVR-De-Echo-Normal",
            "UVR-DeNoise", "UVR-DeNoise-Lite", "mel_band_roformer", "MDX23C", "UVR-MDX-NET",
            "drumsep", "roformer", "viperx", "crowd", "karaoke", "instrumental", "_InstVoc", "_VOCFT",
            "NoReverb", "NoEcho", "NoDelay", "NoCrowd", "NoNoise", "_mel_band_roformer_karaoke_aufr33_viperx_sdr_10"
        ]
        filtered = []
        for g in all_parens:
            if not any(p.lower() in g.lower() for p in to_strip):
                filtered.append(g)

        final_name = base_only + "_" + "".join(filtered) + ext
        final_name = final_name.replace(") (", ")(").replace("__", "_")
        final_path = os.path.join(dirname, final_name)
        if os.path.exists(filepath):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(filepath, final_path)
        return final_path

    @staticmethod
    def _hash_file(filepath: str) -> str:
        """
        Compute a sha256 hash of the file contents.
        Useful for verifying the file hasn't changed between runs.
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
