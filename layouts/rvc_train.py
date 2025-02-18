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

from handlers.args import ArgHandler
from handlers.config import model_path, output_path, app_path
from handlers.download import download_files
from modules.rvc.configs.config import Config
from modules.rvc.infer.modules.train.extract.extract_f0_print import extract_f0_features
from modules.rvc.infer.modules.train.extract.extract_f0_rmvpe import extract_f0_features_rmvpe
from modules.rvc.infer.modules.train.extract.extract_f0_rmvpe_dml import extract_f0_features_rmvpe_dml
from modules.rvc.infer.modules.train.extract_feature_print import extract_feature_print
from modules.rvc.infer.modules.train.preprocess import preprocess_trainset
from modules.rvc.infer.modules.train.train import train_main
from modules.rvc.utils import HParams
from util.data_classes import ProjectFiles
from wrappers.separate import Separate

logger = logging.getLogger(__name__)
now_dir = os.getcwd()
sys.path.append(now_dir)

config = Config()
F0GPUVisible = config.dml is False

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
    vocal_outputs = [output for output in output_files if '(Vocals)' in output and "(BG_Vocals" not in output]
    bg_vocal_outputs = [output for output in output_files if '(BG Vocals' in output]
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


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, progress: gr.Progress):
    sr = sr_dict[sr]
    per = config.preprocess_per if hasattr(config, "preprocess_per") else 3.0
    return preprocess_trainset(trainset_dir, sr, n_p, exp_dir, per, progress)


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
        resume_training,
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
        progress=gr.Progress(),
):
    config = Config()
    exp_dir = os.path.join(output_path, "voices", voice_name)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if model_version == "v1" else "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")

    def get_basenames(directory: str) -> set:
        """Return a set of file basenames (up to the first dot) from a directory."""
        return {filename.split('.')[0] for filename in os.listdir(directory)}

    def fix_escapes(entries: List[str]) -> List[str]:
        """Fix backslashes in file paths."""
        return [entry.replace("\\", "\\\\") for entry in entries]

    def build_entry(name: str, use_pitch_guidance: bool, gt_wavs_dir: str, feature_dir: str,
                    f0_dir: str, f0nsf_dir: str, speaker_index: int) -> str:
        """Build a single entry string based on the provided file name."""
        if use_pitch_guidance:
            wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
            feature_path = os.path.join(feature_dir, f"{name}.npy")
            f0_path = os.path.join(f0_dir, f"{name}.wav.npy")
            f0nsf_path = os.path.join(f0nsf_dir, f"{name}.wav.npy")
            return f"{wav_path}|{feature_path}|{f0_path}|{f0nsf_path}|{speaker_index}"
        else:
            wav_path = os.path.join(gt_wavs_dir, f"{name}.wav")
            feature_path = os.path.join(feature_dir, f"{name}.npy")
            return f"{wav_path}|{feature_path}|{speaker_index}"

    # Get the intersection of base names from the required directories
    if use_pitch_guidance:
        f_names = (
                get_basenames(gt_wavs_dir) &
                get_basenames(feature_dir) &
                get_basenames(f0_dir) &
                get_basenames(f0nsf_dir)
        )
    else:
        f_names = get_basenames(gt_wavs_dir) & get_basenames(feature_dir)

    # Build the list of entries for each file name
    opt = [build_entry(f_name, use_pitch_guidance, gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir, speaker_index)
           for f_name in f_names]

    # Determine feature dimension and mute paths
    fea_dim = 256 if model_version == "v1" else 768
    mutes_dir = os.path.join(app_path, "modules", "rvc")

    mute_wav_path = os.path.join(mutes_dir, "logs", "mute", "0_gt_wavs")
    mute_feature_path = os.path.join(mutes_dir, "logs", "mute", f"3_feature{fea_dim}")
    mute_f0_path = os.path.join(mutes_dir, "logs", "mute", "2a_f0")
    mute_f0nsf_path = os.path.join(mutes_dir, "logs", "mute", "2b-f0nsf")

    # Append mute entries twice to the list
    for _ in range(2):
        if use_pitch_guidance:
            mute_entry = "|".join([
                os.path.join(mute_wav_path, f"mute{sample_rate}.wav"),
                os.path.join(mute_feature_path, "mute.npy"),
                os.path.join(mute_f0_path, "mute.wav.npy"),
                os.path.join(mute_f0nsf_path, "mute.wav.npy"),
                str(speaker_index)
            ])
        else:
            mute_entry = "|".join([
                os.path.join(mute_wav_path, f"mute{sample_rate}.wav"),
                os.path.join(mute_feature_path, f"{fea_dim}", "mute.npy"),
                str(speaker_index)
            ])
        opt.append(mute_entry)

    shuffle(opt)
    opt = fix_escapes(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(more_gpu_ids))
    # If resume_training is True, last_epoch should be the last epoch number.
    if resume_training:
        existing_pretrained_dir = os.path.join(exp_dir, "checkpoints")
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
    hparams.train.epochs = total_epochs
    hparams.sample_rate = sample_rate
    hparams.if_f0 = 1 if use_pitch_guidance else 0
    hparams.if_latest = 1 if save_latest_only else 0
    hparams.save_every_weights = 1 if save_weights_each_ckpt == "Yes" else 0
    hparams.if_cache_data_in_gpu = 1 if cache_dataset_to_gpu == "Yes" else 0
    hparams.data.training_files = os.path.join(exp_dir, "filelist.txt")

    # Logging for debugging.
    logger.info(f"Training with hparams: {hparams}")

    # Directly call the main() function from train script.
    train_main(hparams, progress)
    return "Training complete."


def train_index(project_name, model_version):
    exp_dir = os.path.join(output_path, "voices", project_name)
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
        print(f"Copying index to {index_root}")
        shutil.copy(added_index_path, os.path.join(index_root, f"{os.path.basename(exp_dir)}.index"))
        infos.append(f"Linked index to external location: {outside_index_root}")
    except Exception:
        infos.append(f"Failed to link index to external location: {outside_index_root}")

    yield "\n".join(infos)


def train1key(
        project_name,
        existing_project_name,
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
        progress=gr.Progress()
):
    infos = []
    resuming_training = False
    if (not project_name or project_name == "") and (not existing_project_name or existing_project_name == ""):
        return "Please provide a project name."
    if project_name and existing_project_name:
        return "Please provide only one project name."

    if (not project_name or project_name == "") and existing_project_name:
        logger.info("Using existing project name")
        project_name = existing_project_name
        resuming_training = True
        input_dir = os.path.join(output_path, "voices", project_name, "raw")
        if not os.path.exists(input_dir):
            return f"Project {project_name} does not exist. Please provide a valid project name."
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        inputs = input_files

    if not inputs:
        return "Please provide input files."

    def get_info_str(strr):
        infos.append(strr)
        logger.info(strr)
        return "\n".join(infos)

    exp_dir = os.path.join(output_path, "voices", project_name)
    data_dir = os.path.join(output_path, "voices", project_name, "raw")
    os.makedirs(data_dir, exist_ok=True)
    if num_cpus == 0:
        num_cpus = 1

    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if project_version == "v1" else "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    missing_dirs = []
    for d in [gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir, data_dir]:
        if not os.path.exists(d):
            missing_dirs.append(d)
    # Preprocess
    if not resuming_training or len(missing_dirs) > 0:
        yield get_info_str("Step1: Preprocessing data.")
        vocal_files = inputs
        if separate_vocals:
            vocal_files, bg_vocal_files = separate_vocal(inputs, progress)
            yield get_info_str(f"Separated vocals from {len(vocal_files)} files.")

        for index, f in enumerate(vocal_files):
            progress(index / len(vocal_files), f"Processing {f} ({index + 1}/{len(vocal_files)})")
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
                logger.error(f"Error processing file {f}: {e}")
        preprocess_dataset(data_dir, exp_dir, tgt_sample_rate, num_cpus, progress)

        # Extract pitch features
        yield get_info_str("Step2: Extracting pitch features.")
        progress(0.25, "Extracting pitch features.")
        extract_f0_feature(
            num_cpus,
            extraction_method,
            use_pitch_guidance,
            exp_dir,
            project_version,
            gpus_rmvpe
        )
    else:
        progress(0.25, "Skipping pitch feature extraction.")
        yield get_info_str("Step1: Data already preprocessed.")
        yield get_info_str("Step2: Pitch features already extracted.")

    progress(0.5, "Training model.")
    yield get_info_str("Step3: Training model.")
    click_train(
        project_name,
        resuming_training,
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
        progress
    )

    index_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}.index")
    if not os.path.exists(index_file):
        progress(0.75, "Building index")
        yield get_info_str("Step4: Training complete, now building index.")
        [get_info_str(_) for _ in train_index(project_name, project_version)]
    else:
        yield get_info_str("Step4: Index already exists.")

    yield get_info_str("Processing complete!")


def do_train_index(project_name, existing_project, project_version, progress=gr.Progress()):
    if existing_project is not None and existing_project != "":
        project_name = existing_project
    exp_dir = os.path.join(output_path, "voices", project_name)

    infos = []

    def get_info_str(strr):
        infos.append(strr)
        logger.info(strr)
        return "\n".join(infos)

    if not os.path.exists(exp_dir):
        return get_info_str("Project not found.")

    model_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}_final.pth")
    model_file_new = os.path.join(index_root, f"{os.path.basename(exp_dir)}.pth")
    if os.path.exists(model_file_new):
        model_file = model_file_new
    if not os.path.exists(model_file):
        return get_info_str("Model file not found.")
    # Get the vers key from the model file
    with open(model_file, "rb") as f:
        model_dict = torch.load(f)
        vers = model_dict.get("version", "v2")
    index_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}.index")
    progress(0.25, "Building index")
    yield get_info_str("Step4: Training complete, now building index.")
    [get_info_str(_) for _ in train_index(project_name, project_version)]
    progress(1, "Processing complete.")
    yield get_info_str("Processing complete!")


def list_voice_projects():
    output = [""]
    if not os.path.exists(output_path) or not os.path.exists(os.path.join(output_path, "voices")):
        return output
    all_voices = [name for name in os.listdir(os.path.join(output_path, "voices")) if
                  os.path.isdir(os.path.join(output_path, "voices", name))]
    output.extend(all_voices)
    return output


def list_voice_projects_ui():
    return gr.update(choices=list_voice_projects())


def list_project_weights(project_dir):
    if not project_dir or not os.path.exists(project_dir):
        return []
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    return [name for name in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, name))]


def render():
    with gr.Blocks() as rvc_train:
        gr.Markdown("## RVC Training")
        with gr.Row():
            # GR Markdown saying RVC Training and cool emoji
            with gr.Column():
                gr.Markdown("### 🔧 Settings")
                voice_name = gr.Textbox(
                    label="Voice Name",
                    value="",
                    elem_classes="hintitem", elem_id="rvc_voice_name"
                )
                with gr.Row():
                    existing_project = gr.Dropdown(
                        label="Existing Project",
                        choices=list_voice_projects(),
                        value="",
                        elem_classes="hintitem", elem_id="rvc_existing_project"
                    )
                    refresh_button = gr.Button(
                        "Refresh",
                        variant="secondary",
                        elem_classes="hintitem", elem_id="rvc_refresh_button"
                    )
                    refresh_button.click(fn=list_voice_projects_ui, outputs=[existing_project])

                total_epochs = gr.Slider(
                    minimum=2,
                    maximum=1000,
                    step=1,
                    label="Total Training Epochs",
                    value=125,
                    interactive=True,
                    elem_classes="hintitem", elem_id="rvc_total_epochs"
                )

                train_batch_size = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    label="Batch Size per GPU",
                    value=default_batch_size,
                    interactive=True,
                    elem_classes="hintitem", elem_id="rvc_train_batch_size"
                )

                separate_vocals = gr.Checkbox(
                    label="Separate Vocals",
                    value=True,
                    elem_classes="hintitem", elem_id="rvc_separate_vocals"
                )

                with gr.Accordion(label="Advanced", open=False):
                    sample_rate = gr.Radio(
                        label="Target Sampling Rate",
                        choices=["40k", "48k"],
                        value="48k",
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_sample_rate"
                    )
                    use_pitch_guidance = gr.Radio(
                        label="Use Pitch Guidance",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_pitch_guidance"
                    )
                    model_version = gr.Radio(
                        label="Model Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=True,
                        elem_classes="hintitem", elem_id="rvc_model_version"
                    )
                    num_cpu_processes = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label="Number of CPU Processes",
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        visible=config.n_cpu >= 1,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_num_cpu"
                    )
                    speaker_id = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="Speaker ID",
                        value=0,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_speaker_id"
                    )
                    pitch_extraction_method = gr.Radio(
                        label="Pitch Extraction Method",
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_pitch_extraction"
                    )
                    gpus_rmvpe = gr.Textbox(
                        label="Number of GPUs for RMVPE",
                        value=f"{gpus}-{gpus}",
                        interactive=True,
                        visible=F0GPUVisible,
                        elem_classes="hintitem", elem_id="rvc_gpus_rmvpe"
                    )
                    pitch_extraction_method.change(
                        fn=change_f0_method,
                        inputs=[pitch_extraction_method],
                        outputs=[gpus_rmvpe],
                    )
                    save_epoch_frequency = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label="Save Checkpoint Frequency (Epochs)",
                        value=25,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_save_epoch_frequency"
                    )
                    save_latest_only = gr.Radio(
                        label="Save Only Latest Checkpoint",
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_save_latest_only"
                    )
                    cache_dataset_to_gpu = gr.Radio(
                        label="Cache Dataset to GPU",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_cache_dataset"
                    )
                    save_weights_each_ckpt = gr.Radio(
                        label="Save Weights at Each Checkpoint",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_save_weights"
                    )
                    pretrained_generator = gr.Textbox(
                        label="Pretrained Generator Path",
                        value=os.path.join(model_path, "rvc", "pretrained_v2", "f0G48k.pth"),
                        interactive=True,
                        visible=False,
                        elem_classes="hintitem", elem_id="rvc_pretrained_generator"
                    )
                    pretrained_discriminator = gr.Textbox(
                        label="Pretrained Discriminator Path",
                        value=os.path.join(model_path, "rvc", "pretrained_v2", "f0D48k.pth"),
                        interactive=True,
                        visible=False,
                        elem_classes="hintitem", elem_id="rvc_pretrained_discriminator"
                    )
                    more_gpu_ids = gr.Textbox(
                        label="GPU IDs (e.g., 0-1-2)",
                        value=gpus,
                        interactive=True,
                        elem_classes="hintitem", elem_id="rvc_gpu_ids"
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
            with gr.Column():
                gr.Markdown("### 🎤 Input")
                input_files = gr.File(
                    label="Input Files",
                    type="filepath",
                    file_types=["audio"],
                    file_count="multiple"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        input_url = gr.Textbox(label='Input URL', placeholder='Enter URL', visible=True,
                                               interactive=True, key="process_input_url")
                    with gr.Column():
                        input_url_button = gr.Button(value='Load', visible=True, interactive=True)
            with gr.Column():
                gr.Markdown("### 🎶 Outputs")
                with gr.Row():
                    start_train = gr.Button(
                        "Train",
                        elem_classes="hintitem", elem_id="rvc_start_train", variant="primary"
                    )
                    train_index = gr.Button(
                        "Build Index",
                        elem_classes="hintitem", elem_id="rvc_train_index", variant="secondary",
                    )
                    cancel_train = gr.Button(
                        "Cancel",
                        variant="secondary",
                        visible=False,
                        elem_classes="hintitem", elem_id="rvc_cancel_train"
                    )

                info3 = gr.Textbox(
                    label="Output Info",
                    value="",
                    max_lines=10,
                    elem_classes="hintitem", elem_id="rvc_output_info"
                )
                start_train.click(
                    train1key,
                    [
                        voice_name,
                        existing_project,
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
                        gpus_rmvpe
                    ],
                    info3,
                    api_name="train_start_all",
                )

                train_index.click(
                    do_train_index,
                    [voice_name, existing_project, model_version],
                    info3,
                )

            def update_time_info(input_files):
                yield gr.update(value=f"Calculating length of {len(input_files)} input files...")
                total_length = 0
                if not input_files:
                    yield gr.update(value="")
                for f in input_files:
                    try:
                        audio = AudioSegment.from_file(f)
                        total_length += len(audio) / 1000
                        yield gr.update(
                            value=f"Total length: {total_length / 60:.2f} minutes. \nRecommended is 30-60 minutes.")
                    except Exception as e:
                        logger.error(f"Error processing file {f}: {e}")
                total_length /= 60
                info_value = f"Total length of input files: {total_length:.2f} minutes.\nRecommended is 30-60 minutes."
                yield gr.update(value=info_value)

            input_files.change(
                fn=update_time_info,
                inputs=[input_files],
                outputs=[info3],
            )

        input_url_button.click(
            fn=download_files,
            inputs=[input_url, input_files],
            outputs=[input_files]
        )
    return rvc_train


def register_descriptions(arg_handler: ArgHandler):
    descriptions = {
        "voice_name": "Enter a name for the voice model you are training.",
        "existing_project": "Select an existing project to continue training or leave blank for a new one.",
        "refresh_button": "Click to refresh the list of available voice projects.",
        "total_epochs": "Set the total number of training epochs. More epochs generally improve quality.",
        "train_batch_size": "Adjust the batch size per GPU. Higher values require more VRAM but train faster.",
        "separate_vocals": "Check this box to separate vocals from instrumentals before training.",
        "sample_rate": "Select the target sample rate for the model (40k or 48k).",
        "pitch_guidance": "Choose whether to use pitch guidance for training. Helps with vocal accuracy.",
        "model_version": "Select the RVC model version (v1 or v2).",
        "num_cpu": "Specify the number of CPU processes for data processing and pitch extraction.",
        "speaker_id": "Set the speaker ID to use during training. Some datasets require this.",
        "pitch_extraction": "Select the pitch extraction method to use. RMVPE generally provides the best results.",
        "gpus_rmvpe": "Enter the number of GPUs allocated for RMVPE-based pitch extraction.",
        "save_epoch_frequency": "Set how often (in epochs) to save checkpoints during training.",
        "save_latest_only": "Enable this to keep only the latest checkpoint and save disk space.",
        "cache_dataset": "Choose whether to cache the dataset to GPU memory for faster training.",
        "save_weights": "Enable this to save model weights at every checkpoint.",
        "pretrained_generator": "Path to the pretrained generator model used for fine-tuning.",
        "pretrained_discriminator": "Path to the pretrained discriminator model used for training.",
        "gpu_ids": "Specify GPU IDs for multi-GPU training, separated by dashes (e.g., 0-1-2).",
        "start_train": "Click to begin training the voice model with the selected settings.",
        "cancel_train": "Click to cancel the training process if needed.",
        "output_info": "Displays logs and training progress information.",
        "train_index": "Click to build or re-build the index for the trained voice model.",
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("rvc", elem_id, description)
