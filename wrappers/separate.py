import os
import re
import threading
from typing import Any, List, Dict

import librosa
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator
from scipy import signal

from handlers.config import model_path, output_path
from handlers.reverb import extract_reverb
from modules.audio_separator.audio_separator import separate_music
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Separate(BaseWrapper):
    def register_api_endpoint(self, api) -> Any:
        pass

    title = "Separate"
    priority = 1
    separator = None
    default = True
    description = "Separate audio into distinct stems, such as vocals, instruments, and percussion, as well as remove reverb, echo, delay, crowd noise, and general background noise."
    file_operation_lock = threading.Lock()

    allowed_kwargs = {
        "separate_stems": TypedInput(
            default=False,
            description="Enable to separate the audio into distinct stems, such as vocals, instruments, and percussion. Useful for remixing or isolating specific elements.",
            type=bool,
            gradio_type="Checkbox"
        ),
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
        "store_reverb_ir": TypedInput(
            default=False,
            description="Store the impulse response for reverb removal. This allows for reapplying the reverb later during stem merging. Note: This is an experimental feature and may not always produce accurate results.",
            type=bool,
            gradio_type="Checkbox"
        ),
        "reverb_removal": TypedInput(
            default="Main Vocals",
            description="Choose the scope of reverb removal. Options include leaving it unchanged, applying it to main vocals only, all vocals, or all stems.",
            type=str,
            choices=["Nothing", "Main Vocals", "All Vocals", "All"],
            gradio_type="Dropdown"
        ),
        "echo_removal": TypedInput(
            default="Nothing",
            description="Select the level of echo removal. You can remove echo from nothing, main vocals, all vocals, or all stems.",
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
        "delay_removal_model": TypedInput(
            default="UVR-De-Echo-Normal.pth",
            description="Select the model used for echo and delay removal. Different models offer varying levels of performance and accuracy.",
            type=str,
            choices=[
                "UVR-De-Echo-Normal.pth",
                "UVR-DeEcho-DeReverb.pth"
            ],
            gradio_type="Dropdown"
        ),
        "noise_removal_model": TypedInput(
            default="UVR-DeNoise.pth",
            description="Choose the model used for noise removal. The 'Lite' version may perform faster but with slightly reduced quality.",
            type=str,
            choices=[
                "UVR-DeNoise.pth",
                "UVR-DeNoise-Lite.pth"
            ],
            gradio_type="Dropdown"
        ),
        "crowd_removal_model": TypedInput(
            default="UVR-MDX-NET_Crowd_HQ_1.onnx",
            description="Select the model for removing crowd noise. Different models may excel in specific environments or audio scenarios.",
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

        Returns: [all outputs], [final_instrumental_path, final_vocals_path]
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
        for background_vocal_model in bg_model:
            self.separator.load_model(background_vocal_model)
            files = self.separator.separate(current_path)
            files = self._fix_output_paths(output_dir, files)
            out_files.extend(files)
            v_path = None
            bg_path = None
            for f in files:
                if "(Vocals)_(Vocals)" in f:
                    v_path = f
                elif "(Vocals)_(Instrumental)" in f:
                    bg_path = f

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

        if len(vocals_list) < 3 or len(bg_list) < 3:
            return out_files, []

        ref_len = vocals_list[0].shape[1]
        for arr in vocals_list[1:]:
            if arr.shape[1] < ref_len:
                ref_len = arr.shape[1]

        for i in range(len(vocals_list)):
            vocals_list[i] = vocals_list[i][:, :ref_len]

        stacked_v = np.stack(vocals_list, axis=0)
        avg_vocals = np.mean(stacked_v, axis=0)

        vocals_low = lr_filter(avg_vocals, cutoff=10000, ftype='lowpass', order=6, sr=44100) * 1.01
        vocals_high = lr_filter(vocals_list[1], cutoff=10000, ftype='highpass', order=6, sr=44100)
        final_vocals = vocals_low + vocals_high

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

        os.makedirs(output_dir, exist_ok=True)
        final_vocals_path = os.path.join(output_dir, f"{base_name}_(Vocals).wav")
        final_bg_path = os.path.join(output_dir, f"{base_name}_(BG_Vocals).wav")

        min_len = min(final_vocals.shape[1], final_bg.shape[1])
        final_vocals = final_vocals[:, :min_len]
        final_bg = final_bg[:, :min_len]

        sf.write(final_vocals_path, final_vocals.T, sr, subtype="FLOAT")
        sf.write(final_bg_path, final_bg.T, sr, subtype="FLOAT")
        out_files.extend([final_vocals_path, final_bg_path])
        return out_files, [final_bg_path, final_vocals_path]

    def process_audio(self, inputs: List[ProjectFiles], callback=None, **kwargs: Dict[str, Any]) -> List[ProjectFiles]:
        self.setup()

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        bg_vocals_removal = filtered_kwargs.get("separate_bg_vocals", True)
        reverb_removal = filtered_kwargs.get("reverb_removal", "Vocals")
        echo_removal = filtered_kwargs.get("echo_removal", "Nothing")
        delay_removal = filtered_kwargs.get("delay_removal", "Nothing")
        crowd_removal = filtered_kwargs.get("crowd_removal", "All")
        noise_removal = filtered_kwargs.get("noise_removal", "All")
        delete_extra_stems = filtered_kwargs.get("delete_extra_stems", True)
        store_reverb_ir = filtered_kwargs.get("store_reverb_ir", False)

        delay_removal_model = filtered_kwargs.get("delay_removal_model", "UVR-De-Echo-Normal.pth")
        noise_removal_model = filtered_kwargs.get("noise_removal_model", "UVR-DeNoise.pth")
        crowd_removal_model = filtered_kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")

        separate_stems = filtered_kwargs.get("separate_stems", False)

        all_outputs = []  # Everything, including intermediate steps
        outputs = []  # The ones we want
        pj_outputs = []
        for project in inputs:
            project_outputs = []
            input_file = project.src_file
            output_dir = os.path.join(project.project_dir, "stems")
            self.separator.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            original_basename = os.path.splitext(os.path.basename(input_file))[0]

            separated_stems = separate_music(
                [input_file], output_dir, cpu=False, overlap_demucs=0.1, overlap_VOCFT=0.1,
                overlap_VitLarge=1, overlap_InstVoc=1, weight_InstVoc=8, weight_VOCFT=1,
                weight_VitLarge=5, single_onnx=False, large_gpu=True, BigShifts=7,
                vocals_only=not separate_stems, use_VOCFT=True, output_format="FLOAT", callback=callback
            )

            print(f"Separated stems: {separated_stems}")
            all_outputs.extend(separated_stems)

            vocal_stems = [stem_path for stem_path in separated_stems if "(Vocals)" in stem_path]
            stems_to_filter = [stem_path for stem_path in separated_stems if "(Vocals)" not in stem_path]

            for stem_path in vocal_stems:
                current_path = stem_path
                if bg_vocals_removal:
                    all_vox, out_files = self.separate_bg_multi(current_path, output_dir)
                    all_outputs.extend(all_vox)
                    stems_to_filter.extend(out_files)
                else:
                    all_outputs.append(current_path)
                    stems_to_filter.append(current_path)

            transformations = [
                ("Reverb_HQ_By_FoxJoy.onnx", "No Reverb", reverb_removal),
                (delay_removal_model, "No Echo", echo_removal),
                (delay_removal_model, "No Delay", delay_removal),
                (crowd_removal_model, "No Crowd", crowd_removal),
                (noise_removal_model, "No Noise", noise_removal),
            ]

            for stem_path in stems_to_filter:
                current_path = stem_path
                stem_basename = os.path.basename(current_path)

                for model, tgt_file, transform_flag in transformations:
                    if self._should_apply_transform(stem_basename, transform_flag):
                        self.separator.load_model(model)
                        out_files = self.separator.separate(current_path)
                        out_files = self._fix_output_paths(output_dir, out_files)
                        all_outputs.extend(out_files)
                        file_idx = 0 if tgt_file in out_files[0] else 1
                        current_path = self._rename_file(original_basename, out_files[file_idx])
                        if tgt_file == "No Reverb" and "(Vocals)" in stem_basename and store_reverb_ir:
                            print(f"Extracting IR for {current_path}")
                            reverb_path_idx = 1 if file_idx == 0 else 0
                            reverb_file = out_files[reverb_path_idx]
                            ir_output_path = os.path.join(project.project_dir, "impulse_response.ir")
                            try:
                                impulse_response = extract_reverb(current_path, reverb_file, ir_output_path)
                                print(f"Extracted IR: {impulse_response}")
                            except Exception as e:
                                print(f"Error extracting IR: {e}")
                                impulse_response = None

                project_outputs.append(current_path)
            project.add_output("stems", project_outputs)
            outputs.extend(project_outputs)

            pj_outputs.append(project)
        if delete_extra_stems:
            for p in all_outputs:
                if p not in outputs and os.path.exists(p):
                    self.del_stem(p)

        return pj_outputs

    def del_stem(self, stem_path: str) -> bool:
        try:
            with self.file_operation_lock:
                if os.path.exists(stem_path):
                    os.remove(stem_path)
                    print(f"Deleted: {stem_path}")
                    return not os.path.exists(stem_path)
                return False
        except Exception as e:
            print(f"Error deleting {stem_path}: {e}")
            return False

    def _fix_output_paths(self, output_dir: str, filenames: List[str]) -> List[str]:
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
        elif setting == "All Vocals":
            return self._is_vocal_stem(stem_name)
        elif setting == "Main Vocals":
            return self._is_vocal_stem(stem_name) and "(BG_Vocals)" not in stem_name
        return False

    def _is_vocal_stem(self, name: str) -> bool:
        """
        Simple check to see if it's a "vocal" stem.
        We look for '(Vocals)' or 'Vocal' in the name
        """
        return "(Vocals)" in name or "(BG_Vocals)" in name
