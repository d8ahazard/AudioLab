import os
import re
from typing import Any, List, Dict

import librosa
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
from scipy import signal

from handlers.config import model_path, output_path
from modules.audio_separator.audio_separator import separate_music
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Separate(BaseWrapper):
    title = "Separate"
    priority = 1
    separator = None
    default = True

    allowed_kwargs = {
        "separate_stems": TypedInput(
            default=False,
            description="Whether to separate the audio into instrument stems.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "bg_vocals_removal": TypedInput(
            default="Vocals",
            description="Remove background vocals: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "reverb_removal": TypedInput(
            default="Nothing",
            description="Remove reverb: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "echo_removal": TypedInput(
            default="Nothing",
            description="Remove echo: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "delay_removal": TypedInput(
            default="Nothing",
            description="Remove delay: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "crowd_removal": TypedInput(
            default="Nothing",
            description="Remove crowd noise: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "noise_removal": TypedInput(
            default="Nothing",
            description="Remove general noise: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "Vocals", "All"],
            gradio_type="Dropdown"
        ),

        # Model choices
        "delay_removal_model": TypedInput(
            default="UVR-De-Echo-Normal.pth",
            description="The model used to remove echo/delay.",
            type=str,
            choices=[
                "UVR-De-Echo-Normal.pth",
                "UVR-DeEcho-DeReverb.pth",
            ],
            gradio_type="Dropdown"
        ),
        "noise_removal_model": TypedInput(
            default="UVR-DeNoise.pth",
            description="The model used to remove noise.",
            type=str,
            choices=[
                "UVR-DeNoise.pth",
                "UVR-DeNoise-Lite.pth"
            ],
            gradio_type="Dropdown"
        ),
        "crowd_removal_model": TypedInput(
            default="UVR-MDX-NET_Crowd_HQ_1.onnx",
            description="The model used to remove crowd noise.",
            type=str,
            choices=[
                "UVR-MDX-NET_Crowd_HQ_1.onnx",
                "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
            ],
            gradio_type="Dropdown"
        ),
    }

    def setup(self):
        model_dir = os.path.join(model_path, "audio_separator")
        os.makedirs(model_dir, exist_ok=True)
        output_dir = os.path.join(output_path, "audio_separator")
        os.makedirs(output_dir, exist_ok=True)

        self.separator = Separator(model_file_dir=model_dir, output_dir=output_dir)

        required_models = [
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "Reverb_HQ_By_FoxJoy.onnx",
            "UVR-De-Echo-Normal.pth",
            "UVR-DeEcho-DeReverb.pth",
            "UVR_MDXNET_KARA.onnx",
            "UVR_MDXNET_KARA_2.onnx",
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "UVR-MDX-NET_Crowd_HQ_1.onnx",
            "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
            "htdemucs_6s.yaml",
            "UVR-DeNoise.pth",
            "UVR-DeNoise-Lite.pth"
        ]
        for model in required_models:
            self.separator.download_model_files(model)

    def separate_bg_multi(self, current_path, output_dir):
        """
        Run 3 background-vocal models on the same input track, then combine
        their outputs into a single final Vocals and Instrumental stem via
        simple ensemble averaging + lowpass/highpass filtering.

        Returns: [final_instrumental_path, final_vocals_path]
        """

        # Simple Linkwitz-Riley style filter. Adjust cutoff and order as desired.
        def lr_filter(audio, cutoff, ftype, order=6, sr=44100):
            """
            audio shape: (channels, samples)
            cutoff: frequency in Hz
            ftype: 'lowpass' or 'highpass'
            order: 6 or 4, etc. (but we do order // 2 for actual Butter)
            sr: sample rate
            """
            nyquist = 0.5 * sr
            norm_cutoff = cutoff / nyquist
            b, a = signal.butter(order // 2, norm_cutoff, btype=ftype, analog=False, output='ba')
            sos = signal.tf2sos(b, a)
            filtered = signal.sosfiltfilt(sos, audio, axis=1)
            return filtered

        # Grab a base name for final output
        base_name = os.path.splitext(os.path.basename(current_path))[0]

        # 1) Use each model to separate Vocals & Instrumental.
        bg_model = [
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "UVR_MDXNET_KARA_2.onnx",
            "UVR_MDXNET_KARA.onnx"
        ]

        # We'll store the resulting vocals/instrumental from each model
        # as np arrays in lists:
        vocals_list = []
        bg_list = []
        out_files = []
        sr = 44100
        # Run each background-vocal model
        for background_vocal_model in bg_model:
            self.separator.load_model(background_vocal_model)
            # This returns a list of output file paths (one for vocals, one for instrumental, presumably)
            out_files = self.separator.separate(current_path)
            # We rename them to final folder, but more importantly we want to load them here and store as arrays
            out_files = self._fix_output_paths(output_dir, out_files)

            # Figure out which is vocals vs instrumental, load them
            # (If you know the exact naming, you can parse by substring)
            v_path = None
            bg_path = None
            for f in out_files:
                if "(Vocals)_(Vocals)" in f:
                    v_path = f
                elif "(Vocals)_(Instrumental)" in f:
                    bg_path = f

            # Load them if found
            if v_path and os.path.exists(v_path):
                v_data, sr = librosa.load(v_path, mono=False, sr=44100)
                if len(v_data.shape) == 1:
                    v_data = np.stack([v_data, v_data], axis=0)
                vocals_list.append(v_data)
            if bg_path and os.path.exists(bg_path):
                bg_data, sr = librosa.load(bg_path, mono=False, sr=44100)
                if len(bg_data.shape) == 1:
                    bg_data = np.stack([bg_data, bg_data], axis=0)
                bg_list.append(bg_data)

        # If for some reason we don't have 3 sets, bail out
        if len(vocals_list) < 3 or len(bg_list) < 3:
            # fallback: return the combined paths if you want
            return out_files

        # 2) Combine/ensemble the 3 vocals, removing noise or artifacts:
        # First, match shapes across each array to the shortest of the bunch (or do the longest).
        # We'll pick the first as reference and force all to that shape.
        ref_len = vocals_list[0].shape[1]
        for arr in vocals_list[1:]:
            if arr.shape[1] < ref_len:
                ref_len = arr.shape[1]

        # Now, unify shape
        for i in range(len(vocals_list)):
            vocals_list[i] = vocals_list[i][:, :ref_len]

        # Weighted average approach, plus a lowpass filter and a separate highpass from one model:
        # (Here we just do equal weights. Tweak as you wish.)
        stacked_v = np.stack(vocals_list, axis=0)  # shape: (3, channels, samples)
        avg_vocals = np.mean(stacked_v, axis=0)  # shape: (channels, samples)

        # For example: lowpass the average, highpass from model #2
        vocals_low = lr_filter(avg_vocals, cutoff=10000, ftype='lowpass', order=6, sr=44100) * 1.01
        vocals_high = lr_filter(vocals_list[1], cutoff=10000, ftype='highpass', order=6, sr=44100)
        final_vocals = vocals_low + vocals_high

        # 3) Combine instrumentals similarly
        ref_len_i = bg_list[0].shape[1]
        for arr in bg_list[1:]:
            if arr.shape[1] < ref_len_i:
                ref_len_i = arr.shape[1]
        for i in range(len(bg_list)):
            bg_list[i] = bg_list[i][:, :ref_len_i]
        stacked_i = np.stack(bg_list, axis=0)
        avg_bg = np.mean(stacked_i, axis=0)

        bg_low = lr_filter(avg_bg, 8000, 'lowpass', order=6, sr=44100)
        bg_high = lr_filter(bg_list[2], 8000, 'highpass', order=6, sr=44100)
        final_bg = bg_low + bg_high

        # 4) Write the final tracks
        os.makedirs(output_dir, exist_ok=True)
        final_vocals_path = os.path.join(output_dir, f"{base_name}_(Vocals).wav")
        final_bg_path = os.path.join(output_dir, f"{base_name}_(BG_Vocals).wav")

        # Make sure both final stems are the same length (pick min)
        min_len = min(final_vocals.shape[1], final_bg.shape[1])
        final_vocals = final_vocals[:, :min_len]
        final_bg = final_bg[:, :min_len]

        # Transpose them back to (samples, channels) for soundfile
        sf.write(final_vocals_path, final_vocals.T, sr, subtype="FLOAT")
        sf.write(final_bg_path, final_bg.T, sr, subtype="FLOAT")
        print(f"Saved: {final_vocals_path}, {final_bg_path}")
        # Delete the original files
        for f in out_files:
            if os.path.exists(f):
                try:
                    print(f"Removing: {f}")
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing {f}: {e}")

        # 5) Return only these two final paths
        return [final_bg_path, final_vocals_path]

    def process_audio(self, inputs: List[str], callback=None, **kwargs: Dict[str, Any]) -> List[str]:
        """
        Main pipeline for processing audio according to user-specified transformations.
        """
        self.setup()

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        # Extract transform modes
        bg_vocals_removal = filtered_kwargs.get("bg_vocals_removal", "Vocals")
        reverb_removal = filtered_kwargs.get("reverb_removal", "Vocals")
        echo_removal = filtered_kwargs.get("echo_removal", "Nothing")
        delay_removal = filtered_kwargs.get("delay_removal", "Nothing")
        crowd_removal = filtered_kwargs.get("crowd_removal", "All")
        noise_removal = filtered_kwargs.get("noise_removal", "All")

        # Extract models
        delay_removal_model = filtered_kwargs.get("delay_removal_model", "UVR-De-Echo-Normal.pth")
        noise_removal_model = filtered_kwargs.get("noise_removal_model", "UVR-DeNoise.pth")
        crowd_removal_model = filtered_kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")

        # Separate stems or not
        separate_stems = filtered_kwargs.get("separate_stems", False)

        filtered_inputs, outputs = self.filter_inputs(inputs, "audio")
        output_dir = os.path.join(output_path, "audio_separator")

        for input_file in filtered_inputs:
            original_basename = os.path.splitext(os.path.basename(input_file))[0]

            separated_stems = separate_music(
                [input_file], output_dir, cpu=False, overlap_demucs=0.1, overlap_VOCFT=0.1,
                overlap_VitLarge=1, overlap_InstVoc=1, weight_InstVoc=8, weight_VOCFT=1,
                weight_VitLarge=5, single_onnx=False, large_gpu=True, BigShifts=7,
                vocals_only=not separate_stems, use_VOCFT=True, output_format="FLOAT", callback=callback
            )

            key_swap_dict = {
                "_bass": "(Bass)",
                "_drums": "(Drums)",
                "_other": "(Other)",
                "_vocals": "(Vocals)",
                "_instrum2": "(Instrumental 2)",
                "_instrum": "(Instrumental)",
            }

            for key, value in key_swap_dict.items():
                for idx, stem in enumerate(separated_stems):
                    if key in stem:
                        separated_stems[idx] = stem.replace(key, value)
                        # Rename the file
                        if os.path.exists(separated_stems[idx]):
                            os.remove(separated_stems[idx])
                        os.rename(stem, separated_stems[idx])

            # if separate_stems is true, we keep all except the one that ends with _instrum2, otherwise, _vocals and _instrum
            all_stems = [stem for stem in separated_stems]
            if separate_stems:
                separated_stems = [stem for stem in separated_stems if "(Instrumental 2)" not in stem]
            else:
                separated_stems = [stem for stem in separated_stems if
                                   "(Vocals)" in stem or "(Instrumental)" in stem and "(Instrumental 2)" not in stem]

            # Delete stems in All that are not in separated_stems
            for stem in all_stems:
                if stem not in separated_stems:
                    if os.path.exists(stem):
                        os.remove(stem)

            final_stems = []
            all_intermediate_outputs = separated_stems.copy()
            stems_to_filter = [stem_path for stem_path in separated_stems if "(Vocals)" not in stem_path]
            vocal_stems = [stem_path for stem_path in separated_stems if "(Vocals)" in stem_path]
            for stem_path in vocal_stems:
                # We'll track the current path as we apply transformations
                current_path = stem_path
                stem_basename = os.path.basename(current_path)

                # 2) Background Vocals Removal
                if self._should_apply_transform(stem_basename, bg_vocals_removal):
                    out_files = self.separate_bg_multi(current_path, output_dir)
                    all_intermediate_outputs.extend(out_files)
                    stems_to_filter.extend(out_files)

            print(f"Stems to filter: {stems_to_filter}")
            for stem_path in stems_to_filter:
                current_path = stem_path
                stem_basename = os.path.basename(current_path)

                # 3) Reverb Removal
                if self._should_apply_transform(stem_basename, reverb_removal):
                    self.separator.load_model("Reverb_HQ_By_FoxJoy.onnx")
                    tgt_file = "No Reverb"
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)
                    file_idx = 0 if tgt_file in out_files[0] else 1
                    current_path = self._rename_file(
                        original_basename, out_files[file_idx]
                    )

                # 4) Echo Removal
                if self._should_apply_transform(stem_basename, echo_removal):
                    self.separator.load_model(delay_removal_model)
                    tgt_file = "No Echo"
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)
                    file_idx = 0 if tgt_file in out_files[0] else 1
                    current_path = self._rename_file(
                        original_basename, out_files[file_idx]
                    )

                # 5) Delay Removal
                if self._should_apply_transform(stem_basename, delay_removal):
                    self.separator.load_model(delay_removal_model)
                    tgt_file = "No Delay"
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)
                    file_idx = 0 if tgt_file in out_files[0] else 1
                    current_path = self._rename_file(
                        original_basename, out_files[file_idx]
                    )

                # 6) Crowd Removal
                if self._should_apply_transform(stem_basename, crowd_removal):
                    self.separator.load_model(crowd_removal_model)
                    tgt_file = "No Crowd"
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)
                    file_idx = 0 if tgt_file in out_files[0] else 1
                    current_path = self._rename_file(
                        original_basename, out_files[file_idx]
                    )

                # 7) Noise Removal
                if self._should_apply_transform(stem_basename, noise_removal):
                    self.separator.load_model(noise_removal_model)
                    tgt_file = "No Noise"
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)
                    file_idx = 0 if tgt_file in out_files[0] else 1
                    current_path = self._rename_file(
                        original_basename, out_files[file_idx]
                    )

                final_stems.append(current_path)

            # The final stems for this input_file are appended to `outputs`
            outputs.extend(final_stems)

            for p in all_intermediate_outputs:
                if p not in outputs and os.path.exists(p):
                    print(f"Removing: {p}")
                    try:
                        os.remove(p)
                    except Exception as e:
                        print(f"Error removing {p}: {e}")
        print(f"Final outputs: {outputs}")
        return outputs

    @staticmethod
    def _fix_output_paths(output_dir: str, filenames: List[str]) -> List[str]:
        """Ensure all filenames have the full output path."""
        return [
            os.path.join(output_dir, os.path.basename(filename))
            for filename in filenames
        ]

    def _rename_file(
            self,
            original_base: str,
            filepath: str
    ) -> str:
        """
        Rebuild the filename as:
            <original_base> <any existing parentheses> <new step in parentheses>.ext

        1) Extract existing parentheses from `filepath` (like '(Vocals)', '(No Reverb)', etc.).
        2) Remove anything that's a known model reference or random underscores.
        3) Append `step` as a new parentheses group.
        4) Construct final:  "<original_base> (Vocals)(No Reverb)(Step).ext"
        """

        dirname = os.path.dirname(filepath)
        ext = os.path.splitext(filepath)[1]

        # 1) Find all parentheses in the current filename
        #    E.g. "SongName (Vocals)(No Reverb)_UVRblah => collect '(Vocals)', '(No Reverb)'
        all_parens = re.findall(r'\([^)]*\)', os.path.basename(filepath))
        # This captures each "( ... )" group from the entire basename

        # 2) Filter out model references or anything obviously undesired
        #    E.g. if you have '(model_bs_roformer...)', remove it entirely
        #    We'll keep only parentheses we guess are "valid" steps or stems
        to_strip = [
            "model_bs_roformer", "UVR-DeEcho-DeReverb", "UVR-De-Echo-Normal",
            "UVR_MDXNET_KARA_2", "UVR_MDXNET_KARA", "Reverb_HQ_By_FoxJoy",
            "UVR-MDX-NET_Crowd_HQ_1", "mel_band_roformer_crowd",
            "UVR-DeNoise", "UVR-DeNoise-Lite", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956"
        ]

        filtered_parens = []
        for group in all_parens:
            # If it contains a known pattern => skip
            if any(pat in group for pat in to_strip):
                continue
            filtered_parens.append(group)

        # Remove duplicates if they occur
        # e.g. if somehow "(No Reverb)" was already in the list
        unique_parens = []
        for p in filtered_parens:
            if p not in unique_parens:
                unique_parens.append(p)

        # 4) Build final name:
        #    "original_base + each parenthesis + extension"
        final_name = original_base.strip() + "_" + "".join(unique_parens) + ext
        final_path = os.path.join(dirname, final_name)
        # Replace )_( with )(
        final_path = final_path.replace(") (", ")(")

        # 5) Rename on disk
        if os.path.exists(filepath):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(filepath, final_path)

        return final_path

    def _should_apply_transform(self, stem_name: str, setting: str) -> bool:
        """
        Returns True if we should apply a transform to this stem,
        based on whether `setting` == "All" or "Vocals".
        """
        if setting == "Nothing":
            return False
        elif setting == "All":
            return True
        elif setting == "Vocals":
            return self._is_vocal_stem(stem_name)
        return False

    def _is_vocal_stem(self, name: str) -> bool:
        """
        Simple check to see if it's a "vocal" stem.
        We look for '(Vocals)' or 'Vocal' in the name
        """
        return ("(Vocals)" in name or "(BG_Vocals)") and not "(Instrumental)" in name

    def register_api_endpoint(self, api) -> Any:
        pass

    def del_stem(self, stem_path: str) -> bool:
        if os.path.exists(stem_path):
            os.remove(stem_path)
            return not os.path.exists(stem_path)
        return False
