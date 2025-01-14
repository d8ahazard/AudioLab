import os

import gradio as gr

from handlers.config import model_path
from rvc.infer.modules import vc


def change_choices():
    weight_root = os.path.join(model_path, "cloned")
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def render():
    with gr.Blocks():
        with gr.TabItem("Model Inference"):
            with gr.Row():
                sid0 = gr.Dropdown(label="Inference Timbre", choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        "Refresh Timbre List and Index Path", variant="primary"
                    )
                    clean_button = gr.Button("Unload Timbre to Save Memory", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Select Speaker ID",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem("Single Inference"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label="Pitch Shift (integer, number of semitones, +12 for one octave up, -12 for one octave down)",
                                value=0,
                            )
                            input_audio0 = gr.Textbox(
                                label="Input Audio File Path (default is a correct format example)",
                                placeholder="C:\\Users\\Desktop\\audio_example.wav",
                            )
                            file_index1 = gr.Textbox(
                                label="Feature Search Library File Path (leave empty to use dropdown selection)",
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label="Automatically Detect Index Path (Dropdown)",
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method0 = gr.Radio(
                                label="Select Pitch Extraction Algorithm (use 'pm' for singing to speed up, 'harvest' for better bass but very slow, 'crepe' for good results but GPU-intensive, 'rmvpe' for best results and slightly GPU-intensive)",
                                choices=(
                                    ["pm", "harvest", "crepe", "rmvpe"]
                                    if config.dml == False
                                    else ["pm", "harvest", "rmvpe"]
                                ),
                                value="rmvpe",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label="Post-process Resample to Final Sampling Rate (0 for no resampling)",
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Mix Ratio of Input Source Volume Envelope to Output Volume Envelope (closer to 1 uses more of the output envelope)",
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label="Protect Clear Consonants and Breaths (prevents artifacts like autotune tearing; max 0.5 disables protection, lower values increase protection but may reduce indexing effect)",
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label="Use Median Filtering on 'harvest' Pitch Recognition Results if >=3 (radius determines filtering strength, can reduce mute artifacts)",
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Proportion of Feature Search Used",
                                value=0.75,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label="F0 Curve File (Optional, one pitch per line, replaces default F0 and pitch shift)",
                                visible=False,
                            )

                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                                api_name="infer_refresh",
                            )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button("Convert", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="Output Information")
                            vc_output2 = gr.Audio(
                                label="Output Audio (click the three dots in the bottom right to download)"
                            )

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem("Batch Inference"):
                gr.Markdown(
                    value="Batch conversion: Input a folder of audio files to be converted, or upload multiple audio files. The converted audio will be output to the specified folder (default: opt)."
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label="Pitch Shift (integer, number of semitones, +12 for one octave up, -12 for one octave down)",
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label="Specify Output Folder", value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label="Feature Search Library File Path (leave empty to use dropdown selection)",
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label="Automatically Detect Index Path (Dropdown)",
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label="Select Pitch Extraction Algorithm (use 'pm' for singing to speed up, 'harvest' for better bass but very slow, 'crepe' for good results but GPU-intensive, 'rmvpe' for best results and slightly GPU-intensive)",
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label="Export File Format",
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Post-process Resample to Final Sampling Rate (0 for no resampling)",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Mix Ratio of Input Source Volume Envelope to Output Volume Envelope (closer to 1 uses more of the output envelope)",
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect Clear Consonants and Breaths (prevents artifacts like autotune tearing; max 0.5 disables protection, lower values increase protection but may reduce indexing effect)",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Use Median Filtering on 'harvest' Pitch Recognition Results if >=3 (radius determines filtering strength, can reduce mute artifacts)",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Proportion of Feature Search Used",
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label="Input Folder Path for Audio Files (copy it from the file manager's address bar)",
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label="Alternatively, batch input audio files (either-or, folder takes precedence)",
                    )

                with gr.Row():
                    but1 = gr.Button("Convert", variant="primary")
                    vc_output3 = gr.Textbox(label="Output Information")

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                    api_name="infer_change_voice",
                )
