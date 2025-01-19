import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
from random import shuffle
from time import sleep
from typing import List

import faiss
import gradio as gr
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from sklearn.cluster import MiniBatchKMeans

from handlers.config import model_path, output_path, app_path
from rvc.configs.config import Config
from rvc.infer.modules.train.extract.extract_f0_print import extract_f0_features
from rvc.infer.modules.train.extract.extract_f0_rmvpe import extract_f0_features_rmvpe
from rvc.infer.modules.train.extract.extract_f0_rmvpe_dml import extract_f0_features_rmvpe_dml
from rvc.infer.modules.train.extract_feature_print import extract_feature_print
from rvc.infer.modules.train.preprocess import preprocess_trainset
from rvc.infer.modules.train.train import train_main
from rvc.infer.modules.vc.modules import VC
from rvc.utils import HParams
from util.data_classes import ProjectFiles
from wrappers.separate import Separate

logger = logging.getLogger(__name__)
now_dir = os.getcwd()
sys.path.append(now_dir)

config = Config()
F0GPUVisible = config.dml == False

vc = VC(config)
rvc_path = os.path.join(model_path, "rvc")

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
                value in gpu_name.upper()
                for value in [
                    "10",
                    "16",
                    "20",
                    "30",
                    "40",
                    "A2",
                    "A3",
                    "A4",
                    "P4",
                    "A50",
                    "500",
                    "A60",
                    "70",
                    "80",
                    "90",
                    "M4",
                    "T4",
                    "TITAN",
                    "4060",
                    "L",
                    "6000",
                ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "Unusable GPU, use CPU to extract features."
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

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
    # vc.get_vc(first_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
else:
    first_name = ""
index_paths = []


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        f"{rvc_path}/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        f"{rvc_path}/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            f"{rvc_path}/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            f"{rvc_path}/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            f"{rvc_path}/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def separate_vocal(audio_files: List[str], progress=gr.Progress()) -> List[str]:
    progress(0, f"Separating vocals from {len(audio_files)} audio files...")
    separator = Separate()
    args = {
        "separate_stems": False,
        "remove_bg_vocals": True,
        "reverb_removal": "Main Vocals",
        "echo_removal": "Nothing",
        "delay_removal": "Nothing",
        "crowd_removal": "Nothing",
        "noise_removal": "Nothing",
        "delay_removal_model": "UVR-De-Echo-Normal.pth",
        "noise_removal_model": "UVR-DeNoise.pth",
        "crowd_removal_model": "UVR-MDX-NET_Crowd_HQ_1.onnx",
    }
    project_inputs = []
    for audio_file in audio_files:
        project_inputs.append(ProjectFiles(audio_file))
    outputs = separator.process_audio(project_inputs, progress, **args)
    output_files = []
    for output_project in outputs:
        output_files.extend(output_project.last_outputs)
    vocal_outputs = [output for output in output_files if '(Vocals)' in output and "(BG_Vocals)"]
    bg_vocal_outputs = [output for output in output_files if '(BG Vocals)' in output]
    return vocal_outputs, bg_vocal_outputs


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    per = config.preprocess_per if hasattr(config, "preprocess_per") else 3.0
    return preprocess_trainset(trainset_dir, sr, n_p, exp_dir, per)


def extract_f0_feature(num_processors, extract_method, use_pitch_guidance, exp_dir, project_version, gpus_rmvpe):
    if use_pitch_guidance:
        if extract_method != "rmvpe_gpu":
            extract_f0_features(exp_dir, num_processors, extract_method)
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                # TODO: Threading and shit
                for idx, n_g in enumerate(gpus_rmvpe):
                    extract_f0_features_rmvpe(leng, idx, n_g, exp_dir, config.is_half)
            else:
                extract_f0_features_rmvpe_dml(exp_dir)

    extract_feature_print(config.device, exp_dir, project_version, config.is_half)


def click_train(
        voice_name,
        sample_rate,
        use_pitch_guidance,
        speaker_index,
        save_epoch_frequency,
        total_epochs,
        train_batch_size,
        save_latest_only,
        pretrained_generator,
        pretrained_discriminator,
        more_gpu_ids,
        cache_dataset_to_gpu,
        save_weights_each_ckpt,
        model_version,
):
    config = Config()
    exp_dir = os.path.join(output_path, "voices", voice_name)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if model_version == "v1" else "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    if use_pitch_guidance:
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []

    for name in names:
        if use_pitch_guidance:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    speaker_index,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    speaker_index,
                )
            )
    fea_dim = 256 if model_version == "v1" else 768
    mutes_dir = os.path.join(app_path, "rvc")
    if use_pitch_guidance:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (mutes_dir, sample_rate, mutes_dir, fea_dim, mutes_dir, mutes_dir, speaker_index)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (mutes_dir, sample_rate, mutes_dir, fea_dim, speaker_index)
            )
    shuffle(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(more_gpu_ids))
    if pretrained_generator == "":
        logger.info("No pretrained Generator")
    if pretrained_discriminator == "":
        logger.info("No pretrained Discriminator")
    if model_version == "v1" or sample_rate == "40k":
        config_path = "v1/%s.json" % sample_rate
    else:
        config_path = "v3/%s.json" % sample_rate
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    config_save_path = os.path.join(exp_dir, "config.json")

    # Load configuration.
    with open(config_save_path, "r") as f:
        config = json.load(f)

    # Manually constructing hparams object.
    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = exp_dir
    hparams.save_every_epoch = save_epoch_frequency
    hparams.name = voice_name
    hparams.total_epoch = total_epochs
    hparams.pretrainG = pretrained_generator
    hparams.pretrainD = pretrained_discriminator
    hparams.version = model_version
    hparams.gpus = more_gpu_ids if more_gpu_ids else "0"
    hparams.train.batch_size = train_batch_size
    hparams.sample_rate = sample_rate
    hparams.if_f0 = 1 if use_pitch_guidance else 0
    hparams.if_latest = 1 if save_latest_only == "Yes" else 0
    hparams.save_every_weights = 1 if save_weights_each_ckpt == "Yes" else 0
    hparams.if_cache_data_in_gpu = 1 if cache_dataset_to_gpu == "Yes" else 0
    hparams.data.training_files = f"{exp_dir}/filelist.txt"

    # Logging for debugging.
    logger.info(f"Training with hparams: {hparams}")

    # Directly call the main() function from train script.
    train_main(hparams)
    return "Training complete."


def train_index(exp_dir, model_version):
    os.makedirs(exp_dir, exist_ok=True)

    feature_dir = os.path.join(exp_dir, "3_feature256") if model_version == "v1" else os.path.join(exp_dir,
                                                                                                   "3_feature768")

    if not os.path.exists(feature_dir):
        return "Please extract features first!"

    feature_files = sorted(os.listdir(feature_dir))
    if not feature_files:
        return "Please extract features first!"

    infos = []
    npys = []

    for file_name in feature_files:
        file_path = os.path.join(feature_dir, file_name)
        npys.append(np.load(file_path))

    big_npy = np.concatenate(npys, axis=0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 200000:
        infos.append(f"Trying k-means with {big_npy.shape[0]} data points to 10,000 centers.")
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                ).fit(big_npy).cluster_centers_
            )
        except Exception:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    total_feature_path = os.path.join(exp_dir, "total_fea.npy")
    np.save(total_feature_path, big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"Shape: {big_npy.shape}, IVF size: {n_ivf}")
    yield "\n".join(infos)

    dimension = 256 if model_version == "v1" else 768
    index = faiss.index_factory(dimension, f"IVF{n_ivf},Flat")

    infos.append("Training the index...")
    yield "\n".join(infos)

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    index.train(big_npy)

    trained_index_path = os.path.join(
        exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_version}.index"
    )
    faiss.write_index(index, trained_index_path)

    infos.append("Adding data to the index...")
    yield "\n".join(infos)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])

    added_index_path = os.path.join(
        exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_version}.index"
    )
    faiss.write_index(index, added_index_path)

    infos.append(f"Successfully built index: {added_index_path}")

    try:
        # Copy the file to os.path.join(model_path, "trained", f{voice_name}._index)
        shutil.copy(added_index_path, os.path.join(index_root, f"{os.path.basename(exp_dir)}.index"))
        # external_link = os.link if platform.system() == "Windows" else os.symlink
        # external_index_path = os.path.join(
        #     outside_index_root, f"{os.path.basename(exp_dir)}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_version}.index"
        # )
        # external_link(added_index_path, external_index_path)
        infos.append(f"Linked index to external location: {outside_index_root}")
    except Exception:
        infos.append(f"Failed to link index to external location: {outside_index_root}")

    yield "\n".join(infos)


# voice_name,
# sample_rate,
# use_pitch_guidance,
# input_files,
# speaker_id,
# num_cpu_processes,
# pitch_extraction_method,
# save_epoch_frequency,
# total_epochs,
# train_batch_size,
# save_latest_only,
# pretrained_generator,
# pretrained_discriminator,
# more_gpu_ids,
# cache_dataset_to_gpu,
# save_weights_each_ckpt,
# model_version,
# gpus_rmvpe

def train1key(
        project_name,
        separate_vocals,
        tgt_sample_rate,
        use_pitch_guidance,
        inputs,
        spk_id,
        num_cpus,
        extraction_method,
        epoch_save_freq,
        train_epochs,
        batch_size,
        save_latest,
        generator,
        discriminator,
        tgt_gpus,
        cache_to_gpu,
        save_weights_every,
        project_version,
        gpus_rmvpe,
):
    infos = []

    if not project_name or project_name == "":
        return "Please provide a project name."
    if not inputs:
        return "Please provide input files."
    # Determine the total length in time of input files

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    exp_dir = os.path.join(output_path, "voices", project_name)
    data_dir = os.path.join(output_path, "voices", project_name, "raw")
    os.makedirs(data_dir, exist_ok=True)
    if num_cpus == 0:
        num_cpus = 1
    # Preprocess
    yield get_info_str("Step1: Preprocessing data")
    vocal_files = inputs
    if separate_vocals:
        vocal_files, bg_vocal_files = separate_vocal(inputs)
        yield get_info_str(f"Separated vocals from {len(vocal_files)} files.")

    for f in vocal_files:
        try:
            base_name, ext = os.path.splitext(os.path.basename(f))
            output_file = os.path.join(data_dir, f"{base_name}.wav")

            if ext.lower() != ".wav":
                # Use FFmpeg for conversion
                temp_output = f"{base_name}_temp.wav"
                subprocess.run([
                    "ffmpeg", "-i", f, temp_output, "-y", "-loglevel", "error"
                ], check=True)

                # Ensure the file is readable and has valid data
                audio, samplerate = torchaudio.load(temp_output)
                torchaudio.save(output_file, audio, sample_rate=samplerate)
                os.remove(temp_output)
            else:
                # If already a WAV file, copy it to the data_dir
                shutil.copyfile(f, output_file)

        except Exception as e:
            print(f"Error processing file {f}: {e}")
    preprocess_dataset(data_dir, exp_dir, tgt_sample_rate, num_cpus)

    # Extract pitch features
    yield get_info_str("Step2: Extracting pitch features")
    extract_f0_feature(
        num_cpus,
        extraction_method,
        use_pitch_guidance,
        exp_dir,
        project_version,
        gpus_rmvpe
    )

    # step3a:训练模型
    yield get_info_str("Step3a: Training model")
    click_train(
        project_name,
        tgt_sample_rate,
        use_pitch_guidance,
        spk_id,
        epoch_save_freq,
        train_epochs,
        batch_size,
        save_latest,
        generator,
        discriminator,
        tgt_gpus,
        cache_to_gpu,
        save_weights_every,
        project_version,
    )
    yield get_info_str("Training complete, now building index.")
    # step3b:训练索引
    [get_info_str(_) for _ in train_index(project_name, project_version)]
    yield get_info_str("PRocessing complete.!")


def render():
    with gr.Row():
        but5 = gr.Button("Train", variant="primary")
    with gr.Row():
        with gr.Column():
            voice_name = gr.Textbox(label="Voice Name", value="")
            separate_vocals = gr.Checkbox(label="Separate Vocals", value=True)
            sample_rate = gr.Radio(
                label="Target Sampling Rate",
                choices=["40k", "48k"],
                value="48k",
                interactive=True,
            )
            use_pitch_guidance = gr.Radio(
                label="Use Pitch Guidance",
                choices=[True, False],
                value=True,
                interactive=True,
            )
            model_version = gr.Radio(
                label="Model Version",
                choices=["v1", "v2"],
                value="v2",
                interactive=True,
                visible=True,
            )
            num_cpu_processes = gr.Slider(
                minimum=0,
                maximum=config.n_cpu,
                step=1,
                label="Number of CPU processes for pitch extraction and data processing",
                value=int(np.ceil(config.n_cpu / 1.5)),
                visible=config.n_cpu >= 1,
                interactive=True,
            )
            speaker_id = gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                label="Speaker ID",
                value=0,
                interactive=True,
            )
            pitch_extraction_method = gr.Radio(
                label="Pitch Extraction Method",
                choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                value="rmvpe_gpu",
                interactive=True,
            )
            gpus_rmvpe = gr.Textbox(
                label="Number of GPUs for RMVPE",
                value=f"{gpus}-{gpus}",
                interactive=True,
                visible=F0GPUVisible,
            )
            pitch_extraction_method.change(
                fn=change_f0_method,
                inputs=[pitch_extraction_method],
                outputs=[gpus_rmvpe],
            )
            save_epoch_frequency = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                label="Save Checkpoint Frequency (in epochs)",
                value=5,
                interactive=True,
            )
            total_epochs = gr.Slider(
                minimum=2,
                maximum=1000,
                step=1,
                label="Total Training Epochs",
                value=125,
                interactive=True,
            )
            train_batch_size = gr.Slider(
                minimum=1,
                maximum=40,
                step=1,
                label="Batch Size per GPU",
                value=default_batch_size,
                interactive=True,
            )
            save_latest_only = gr.Radio(
                label="Save Only the Latest Checkpoint to Save Disk Space",
                choices=["Yes", "No"],
                value="No",
                interactive=True,
            )
            cache_dataset_to_gpu = gr.Radio(
                label="Cache Dataset to GPU",
                choices=["Yes", "No"],
                value="Yes",
                interactive=True,
            )
            save_weights_each_ckpt = gr.Radio(
                label="Save Weights at Each Checkpoint",
                choices=["Yes", "No"],
                value="No",
                interactive=True,
            )
            pretrained_generator = gr.Textbox(
                label="Pretrained Generator Path",
                value=os.path.join(model_path, "rvc", "pretrained_v2", "f0G40k.pth"),
                interactive=True,
                visible=False
            )
            pretrained_discriminator = gr.Textbox(
                label="Pretrained Discriminator Path",
                value=os.path.join(model_path, "rvc", "pretrained_v2", "f0D40k.pth"),
                interactive=True,
                visible=False
            )
            sample_rate.change(
                change_sr2,
                [sample_rate, use_pitch_guidance, model_version],
                [pretrained_generator, pretrained_discriminator],
            )
            model_version.change(
                change_version19,
                [sample_rate, use_pitch_guidance, model_version],
                [pretrained_generator, pretrained_discriminator, sample_rate],
            )
            use_pitch_guidance.change(
                change_f0,
                [use_pitch_guidance, sample_rate, model_version],
                [pitch_extraction_method, gpus_rmvpe, pretrained_generator, pretrained_discriminator],
            )
            more_gpu_ids = gr.Textbox(
                label="Enter GPU IDs separated by `-` (e.g., `0-1-2` for GPUs 0, 1, and 2)",
                value=gpus,
                interactive=True,
            )
        with gr.Column():
            input_files = gr.File(label="Input Files", type="filepath", file_types=["audio"], file_count="multiple")
        with gr.Column():
            info3 = gr.Textbox(label="Output Info", value="", max_lines=10)
            but5.click(
                train1key,
                [
                    voice_name,
                    separate_vocals,
                    sample_rate,
                    use_pitch_guidance,
                    input_files,
                    speaker_id,
                    num_cpu_processes,
                    pitch_extraction_method,
                    save_epoch_frequency,
                    total_epochs,
                    train_batch_size,
                    save_latest_only,
                    pretrained_generator,
                    pretrained_discriminator,
                    more_gpu_ids,
                    cache_dataset_to_gpu,
                    save_weights_each_ckpt,
                    model_version,
                    gpus_rmvpe,
                ],
                info3,
                api_name="train_start_all",
            )

        def update_time_info(input_files):
            total_length = 0
            if not input_files:
                return gr.update(value="")
            for f in input_files:
                try:
                    # Load the audio file using pydub
                    audio = AudioSegment.from_file(f)
                    total_length += len(audio) / 1000  # Convert milliseconds to seconds
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
            total_length /= 60  # Convert seconds to minutes
            info_value = f"Total length of input files: {total_length:.2f} minutes."
            info_value += "\nRecommended is 30-60 minutes."
            return gr.update(value=info_value)


        input_files.change(
            fn=update_time_info,
            inputs=[input_files],
            outputs=[info3],
        )
