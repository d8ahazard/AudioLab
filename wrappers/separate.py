import os
import re
from typing import Any, List, Dict

from audio_separator.separator import Separator

from handlers.config import model_path, output_path
from modules.audio_separator.audio_separator import separate_music
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Separate(BaseWrapper):
    title = "Separate"
    priority = 1
    separator = None

    allowed_kwargs = {
        "separate_stems": TypedInput(
            default=False,
            description="Whether to separate the audio into instrument stems.",
            type=bool,
            gradio_type="Checkbox"
        ),

        "bg_vocals_removal": TypedInput(
            default="vocals_only",
            description="Remove background vocals: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
            gradio_type="Dropdown"
        ),
        "reverb_removal": TypedInput(
            default="vocals_only",
            description="Remove reverb: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
            gradio_type="Dropdown"
        ),
        "echo_removal": TypedInput(
            default="Nothing",
            description="Remove echo: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
            gradio_type="Dropdown"
        ),
        "delay_removal": TypedInput(
            default="Nothing",
            description="Remove delay: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
            gradio_type="Dropdown"
        ),
        "crowd_removal": TypedInput(
            default="Nothing",
            description="Remove crowd noise: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
            gradio_type="Dropdown"
        ),
        "noise_removal": TypedInput(
            default="all_stems",
            description="Remove general noise: Nothing, Vocals Only, or All Stems.",
            type=str,
            choices=["Nothing", "vocals_only", "all_stems"],
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
        "background_vocal_model": TypedInput(
            default="UVR_MDXNET_KARA_2.onnx",
            description="The model used to separate background vocals from the main vocals.",
            type=str,
            choices=[
                "UVR_MDXNET_KARA.onnx",
                "UVR_MDXNET_KARA_2.onnx"
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
            "UVR-MDX-NET_Crowd_HQ_1.onnx",
            "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
            "htdemucs_6s.yaml",
            "UVR-DeNoise.pth",
            "UVR-DeNoise-Lite.pth"
        ]
        for model in required_models:
            self.separator.download_model_files(model)

    def process_audio(self, inputs: List[str], callback=None, **kwargs: Dict[str, Any]) -> List[str]:
        """
        Main pipeline for processing audio according to user-specified transformations.
        """
        self.setup()

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}

        # Extract transform modes
        bg_vocals_removal = filtered_kwargs.get("bg_vocals_removal", "vocals_only")
        reverb_removal = filtered_kwargs.get("reverb_removal", "vocals_only")
        echo_removal = filtered_kwargs.get("echo_removal", "Nothing")
        delay_removal = filtered_kwargs.get("delay_removal", "Nothing")
        crowd_removal = filtered_kwargs.get("crowd_removal", "all_stems")
        noise_removal = filtered_kwargs.get("noise_removal", "all_stems")

        # Extract models
        delay_removal_model = filtered_kwargs.get("delay_removal_model", "UVR-De-Echo-Normal.pth")
        background_vocal_model = filtered_kwargs.get("background_vocal_model", "UVR_MDXNET_KARA_2.onnx")
        noise_removal_model = filtered_kwargs.get("noise_removal_model", "UVR-DeNoise.pth")
        crowd_removal_model = filtered_kwargs.get("crowd_removal_model", "UVR-MDX-NET_Crowd_HQ_1.onnx")

        # Separate stems or not
        separate_stems = filtered_kwargs.get("separate_stems", False)

        filtered_inputs, outputs = self.filter_inputs(inputs, "audio")
        output_dir = os.path.join(output_path, "audio_separator")

        for input_file in filtered_inputs:
            original_basename = os.path.splitext(os.path.basename(input_file))[0]
            # e.g. input_file = "/path/to/SongName.wav" => original_base = "SongName"

            # input_audio: List[str], output_folder: str, cpu: bool = False,
            #                    overlap_demucs: float = 0.1, overlap_VOCFT: float = 0.1, overlap_VitLarge: int = 1,
            #                    overlap_InstVoc: int = 1, weight_InstVoc: float = 8, weight_VOCFT: float = 1,
            #                    weight_VitLarge: float = 5, single_onnx: bool = False, large_gpu: bool = False,
            #                    BigShifts: int = 7, vocals_only: bool = False, use_VOCFT: bool = False,
            #                    output_format: str = "FLOAT", callback=None

            separated_stems = separate_music(
                [input_file], output_dir, cpu=False, overlap_demucs=0.1, overlap_VOCFT=0.1,
                overlap_VitLarge=1, overlap_InstVoc=1, weight_InstVoc=8, weight_VOCFT=1,
                weight_VitLarge=5, single_onnx=False, large_gpu=True, BigShifts=7,
                vocals_only=not separate_stems, use_VOCFT=True, output_format="FLOAT", callback=callback
            )
            # 1) Separate either into multiple stems or just vocals/instrumental
            # if separate_stems:
            #     self.separator.load_model("htdemucs_6s.yaml")
            # else:
            #     self.separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")

            # separated_stems = self.separator.separate(input_file)
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
                        os.rename(stem, separated_stems[idx])

            # if separate_stems is true, we keep all except the one that ends with _instrum2, otherwise, _vocals and _instrum
            all_stems = [stem for stem in separated_stems]
            if separate_stems:
                separated_stems = [stem for stem in separated_stems if "(Instrumental 2)" not in stem]
            else:
                separated_stems = [stem for stem in separated_stems if
                                   "(Vocals)" in stem or "(Instrumental)" in stem and "(Instrumental 2)" not in stem]
            # Delete stems in all_stems that are not in separated_stems
            for stem in all_stems:
                if stem not in separated_stems:
                    os.remove(stem)
            # separated_stems = self._fix_output_paths(output_dir, separated_stems)
            # separated_stems = [self._rename_file(original_basename, stem) for stem in separated_stems]

            final_stems = []
            all_intermediate_outputs = []

            for stem_path in separated_stems:
                # We'll track the current path as we apply transformations
                current_path = stem_path
                all_intermediate_outputs.append(current_path)

                stem_basename = os.path.basename(current_path)

                # 2) Background Vocals Removal
                if self._should_apply_transform(stem_basename, bg_vocals_removal):
                    self.separator.load_model(background_vocal_model)
                    out_files = self.separator.separate(current_path)
                    out_files = self._fix_output_paths(output_dir, out_files)
                    all_intermediate_outputs.extend(out_files)

                    # Typically [main_vocals_no_bg, bg_only]
                    if len(out_files) > 1:
                        inst_key = "Instrumental"
                        bg_only_files = [f for f in out_files if inst_key in f]
                        main_vocals_files = [f for f in out_files if inst_key not in f]
                        bg_only = bg_only_files[0] if bg_only_files else None
                        main_vocals = main_vocals_files[0] if main_vocals_files else None
                        # Return the BG voc stem to user as well
                        if bg_only:
                            renamed_bg = self._rename_file(
                                original_basename, bg_only
                            )

                            outputs.append(renamed_bg)
                        # If we have a main vocals stem, we'll use that as the current path
                        if main_vocals:
                            current_path = main_vocals
                    else:
                        # Fallback if only one output
                        current_path = out_files[0]
                if not current_path:
                    continue

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

            #
            # Clean up extraneous intermediate files:
            #
            # all_intermediate_outputs = *all* files created
            # final_stems + BG vocals (some are also in outputs)
            # => We'll remove anything not in outputs
            #
            for path_ in all_intermediate_outputs:
                if path_ not in outputs and os.path.exists(path_):
                    os.remove(path_)

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
        to_strip = {
            "model_bs_roformer", "UVR-DeEcho-DeReverb", "UVR-De-Echo-Normal",
            "UVR_MDXNET_KARA_2", "UVR_MDXNET_KARA", "Reverb_HQ_By_FoxJoy",
            "UVR-MDX-NET_Crowd_HQ_1", "mel_band_roformer_crowd",
            "UVR-DeNoise", "UVR-DeNoise-Lite"
        }

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
        if "(Vocals)(Instrumental)" in final_name:
            final_name = final_name.replace("(Vocals)(Instrumental)", "(Vocals)(BG Vocals)")
        final_path = os.path.join(dirname, final_name)

        # 5) Rename on disk
        if os.path.exists(filepath):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(filepath, final_path)

        return final_path

    def _should_apply_transform(self, stem_name: str, setting: str) -> bool:
        """
        Returns True if we should apply a transform to this stem,
        based on whether `setting` == "all_stems" or "vocals_only".
        """
        if setting == "Nothing":
            return False
        elif setting == "all_stems":
            return True
        elif setting == "vocals_only":
            return self._is_vocal_stem(stem_name)
        return False

    def _is_vocal_stem(self, name: str) -> bool:
        """
        Simple check to see if it's a "vocal" stem.
        We look for '(Vocals)' or 'Vocal' in the name
        """
        return "(Vocals)" in name and not "(Instrumental)" in name

    def register_api_endpoint(self, api) -> Any:
        pass
