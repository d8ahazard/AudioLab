import os
import re
import threading
from typing import Any, List, Dict

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from handlers.config import model_path, output_path
from handlers.reverb import extract_reverb
from modules.audio_separator.audio_separator import separate_music
from util.data_classes import ProjectFiles
from wrappers.base_wrapper import BaseWrapper, TypedInput


class Separate(BaseWrapper):
    """
    An advanced wrapper that:
      1) Separates the audio into multiple stems or just vocals,
      2) Optionally removes reverb/echo/noise/crowd,
      3) Can do background-vocal splitting,
      4) Can do advanced drum separation if requested.

    Uses the new ensemble approach from `ensemble_separator.py` (via separate_music).
    """

    def register_api_endpoint(self, api) -> Any:
        pass

    title = "Separate"
    priority = 1
    separator = None
    default = True
    description = (
        "Separate audio into distinct stems (vocals, instruments, bass, drums, etc.), "
        "optionally remove reverb, echo/delay, crowd noise, and general background noise."
    )
    file_operation_lock = threading.Lock()

    allowed_kwargs = {
        "separate_stems": TypedInput(
            default=False,
            description="Enable to separate the audio into distinct stems, such as vocals, instruments, and percussion. Useful for remixing or isolating specific elements.",
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

        self.separator = Separator(model_file_dir=model_dir, output_dir=out_dir)

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
            files = [os.path.join(output_dir, f) for f in files]
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
        """
        The big method:
          1) Calls `separate_music` to get stems,
          2) Optionally splits background vocals from main,
          3) Then runs reverb/echo/delay/crowd/noise removal in the user-specified way,
          4) Returns final stems.
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

            # 1) Actual separation
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

            # 2) Optionally do BG Vocal splitting (like your old logic).
            #    We'll look for e.g. "...(Vocals).wav", run a karaoke model, rename the "instrumental" => BG_Vocals
            updated_stems = []
            for full_path in separated_stems:
                base_name = os.path.basename(full_path)
                if "(Vocals)" in base_name and separate_bg_vocals:
                    # run a Karaoke model on the vocal track
                    out_files, final_stems = self.separate_bg_multi(full_path, out_dir)
                    all_generated.extend(out_files)
                    updated_stems.extend(final_stems)   # [BG_Vocals, Vocals]
                else:
                    updated_stems.append(full_path)

            # 3) Transform chain: reverb -> echo -> delay -> crowd -> noise
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
                # each transform can produce e.g. "(No Reverb)" and its complementary track
                # we pick the "(No Reverb)" track as the new current_path
                for model_file, out_label, transform_flag in transformations:
                    if self._should_apply_transform(os.path.basename(current_path), transform_flag):
                        self.separator.load_model(model_file)
                        partial = self.separator.separate(current_path)
                        # join
                        partial_full = [os.path.join(project.project_dir, "stems", p) for p in partial]
                        all_generated.extend(partial_full)

                        # we typically want the "NoReverb", "NoEcho", etc. track
                        pick = None
                        alt = None
                        if len(partial_full) == 2:
                            if out_label.replace(" ", "") in partial_full[0]:
                                pick = partial_full[0]
                                alt = partial_full[1]
                            else:
                                pick = partial_full[1]
                                alt = partial_full[0]

                            current_path = self._rename_file(os.path.basename(input_file), pick)
                            # if reverb removal from main vocals, maybe store IR
                            if out_label == "No Reverb" and "(Vocals)" in stem_path and store_reverb_ir and alt:
                                try:
                                    out_ir = os.path.join(project.project_dir, "impulse_response.ir")
                                    extract_reverb(current_path, alt, out_ir)
                                except Exception as e:
                                    print("Error extracting IR:", e)
                        else:
                            # if we have more or fewer than 2 stems
                            # pick whichever has out_label
                            for pf in partial_full:
                                if out_label.replace(" ", "") in pf:
                                    current_path = self._rename_file(os.path.basename(input_file), pf)
                                    break

                final_stem_paths.append(current_path)

            proj_stems.extend(final_stem_paths)
            project.add_output("stems", proj_stems)
            outputs.extend(final_stem_paths)
            projects_out.append(project)

        # 4) Clean up intermediate if requested
        if delete_extra_stems:
            for p in all_generated:
                if p not in outputs and os.path.exists(p):
                    self.del_stem(p)

        return projects_out

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
            return "(Vocals)" in stem_name or "(BG_Vocals)" in stem_name
        if setting == "Main Vocals":
            return "(Vocals)" in stem_name and "(BG_Vocals)" not in stem_name
        return False

    @staticmethod
    def _rename_bgvocal(filepath: str) -> str:
        """
        For the Karaoke sub-separation of main vocals, the 'instrumental' track is actually BG vocals.
        We'll rename the file so it has '(BG_Vocals)' instead of '(Instrumental)'.
        Example: 'SongName_(Instrumental).wav' => 'SongName_(BG_Vocals).wav'
        """
        dirname = os.path.dirname(filepath)
        oldbase = os.path.basename(filepath)
        newbase = oldbase.replace("(Instrumental)", "(BG_Vocals)")
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

        # references to remove
        to_strip = [
            "deverb_bs_roformer", "UVR-DeEcho-DeReverb", "UVR-De-Echo-Normal",
            "UVR-DeNoise", "UVR-DeNoise-Lite", "mel_band_roformer", "MDX23C", "UVR-MDX-NET",
            "drumsep", "roformer", "viperx", "crowd", "karaoke", "instrumental", "_InstVoc", "_VOCFT",
            "NoReverb", "NoEcho", "NoDelay", "NoCrowd", "NoNoise"
        ]
        filtered = []
        for g in all_parens:
            # keep only user-friendly parentheses
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
