import os

import gradio as gr

from handlers.config import model_path
from rvc.configs.config import Config
from rvc.infer.modules.vc.modules import VC

config = Config()
vc = VC(config)

# weight_root = os.getenv("weight_root")
# weight_uvr5_root = os.getenv("weight_uvr5_root")
# index_root = os.getenv("index_root")
# outside_index_root = os.getenv("outside_index_root")

weight_root = os.path.join(model_path, "trained")
weight_uvr5_root = os.path.join(model_path, "trained_onnx")
index_root = os.path.join(model_path, "trained")
outside_index_root = os.path.join(model_path, "outside")

for folder in [weight_root, weight_uvr5_root, index_root, outside_index_root]:
    os.makedirs(folder, exist_ok=True)

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
if len(names):
    first_name = names[0]
    #vc.get_vc(first_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
else:
    first_name = ""
index_paths = []


def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def clean():
    return {"value": "", "__type__": "update"}


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
    return {"choices": sorted(names), "__type__": "update"}


def render():
    with gr.Blocks():
        with gr.Row():
            but1 = gr.Button("Convert", variant="primary")

        with gr.Row():
            with gr.Column():
                sid0 = gr.Dropdown(label="Selected Voice", choices=sorted(names),
                                   value=names[0] if names else "",
                                   interactive=True)
                refresh_button = gr.Button(
                    "Refresh Voice List", variant="primary"
                )
                clean_button = gr.Button("Unload Timbre to Save Memory", variant="primary", visible=False)
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
                vc_transform1 = gr.Number(
                    label="Pitch Shift (integer, number of semitones, +12 for one octave up, -12 for one octave down)",
                    value=0,
                )
                f0method1 = gr.Radio(
                    label="Select Pitch Extraction Algorithm (use 'pm' for singing to speed up, 'harvest' for better bass but very slow, 'crepe' for good results but GPU-intensive, 'rmvpe' for best results and slightly GPU-intensive)",
                    choices=(
                        ["pm", "harvest", "crepe", "rmvpe"]
                        if not config.dml
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
            with gr.Column():
                inputs = gr.File(
                    file_count="multiple",
                    label="Alternatively, batch input audio files (either-or, folder takes precedence)",
                )

            with gr.Column():
                vc_output3 = gr.Textbox(label="Output Information")

            but1.click(
                vc.vc_multi,
                [
                    sid0,
                    spk_item,
                    inputs,
                    vc_transform1,
                    f0method1,
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
                inputs=[sid0, protect1],
                outputs=[spk_item, protect1],
                api_name="infer_change_voice",
            )
