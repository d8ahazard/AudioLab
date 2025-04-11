import json
import logging
import os
import pathlib
import shutil
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
import librosa

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
                    "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500",
                    "A60", "70", "80", "90", "M4", "T4", "TITAN", "4060", "L", "6000",
                ]
        ):
            if_gpu_ok = True  # At least one usable NVIDIA card
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4
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
    generator_path = os.path.join(rvc_path, "pretrained", path_str, f"{f0_str}G{sr2}.pth")
    discriminator_path = os.path.join(rvc_path, "pretrained", path_str, f"{f0_str}D{sr2}.pth")
    if_pretrained_generator_exist = os.path.exists(generator_path)
    if_pretrained_discriminator_exist = os.path.exists(discriminator_path)
    if not if_pretrained_generator_exist:
        logger.error(
            f"{generator_path} not exist, will not use pretrained model."
        )
        raise FileNotFoundError(f"{generator_path} not exist, will not use pretrained model.")
    if not if_pretrained_discriminator_exist:
        logger.error(
            f"{discriminator_path} not exist, will not use pretrained model."
        )
        raise FileNotFoundError(f"{discriminator_path} not exist, will not use pretrained model.")
    return (
        os.path.join(rvc_path, "pretrained", path_str, f"{f0_str}G{sr2}.pth") if if_pretrained_generator_exist else "",
        os.path.join(rvc_path, "pretrained", path_str,
                     f"{f0_str}D{sr2}.pth") if if_pretrained_discriminator_exist else ""
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
        *get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2),
    )


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


def if_done(done, p):
    while True:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while True:
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
                for idx, n_g in enumerate(gpus_rmvpe):
                    extract_f0_features_rmvpe(leng, idx, n_g, exp_dir, config.is_half)
            else:
                extract_f0_features_rmvpe_dml(exp_dir)
    extract_feature_print(config.device, exp_dir, project_version, config.is_half)


def analyze_pitch_range(audio_files, progress=gr.Progress()):
    """Analyze the pitch range of the input audio files."""
    progress(0, "Analyzing pitch range of input files...")
    min_pitch = float('inf')
    max_pitch = float('-inf')
    avg_pitches = []
    
    for idx, file in enumerate(audio_files):
        progress((idx + 1) / len(audio_files), f"Analyzing pitch for file {idx + 1}/{len(audio_files)}")
        try:
            y, sr = librosa.load(file)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Filter out pitches with low magnitude
            pitches_filtered = pitches[magnitudes > np.median(magnitudes)]
            if len(pitches_filtered) > 0:
                file_min = np.min(pitches_filtered)
                file_max = np.max(pitches_filtered)
                file_avg = np.mean(pitches_filtered)
                
                min_pitch = min(min_pitch, file_min)
                max_pitch = max(max_pitch, file_max)
                avg_pitches.append(file_avg)
        except Exception as e:
            logger.error(f"Error analyzing pitch for {file}: {e}")
            continue
    
    if min_pitch == float('inf') or max_pitch == float('-inf'):
        return None
        
    avg_pitch = np.mean(avg_pitches) if avg_pitches else 0
    return {
        'min_pitch': min_pitch,
        'max_pitch': max_pitch,
        'avg_pitch': avg_pitch,
        'pitch_range': max_pitch - min_pitch
    }


def get_dataset_length(index_file):
    """Get the total length of the dataset from the index file."""
    try:
        index = faiss.read_index(index_file)
        return index.ntotal
    except Exception as e:
        logger.error(f"Error reading index file: {e}")
        return 0


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
    try:
        config_obj = Config()
        exp_dir = os.path.join(output_path, "voices", voice_name)
        os.makedirs(exp_dir, exist_ok=True)
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        feature_dir = os.path.join(exp_dir, "3_feature256" if model_version == "v1" else "3_feature768")
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")

        def get_basenames(directory: str) -> set:
            return {filename.split('.')[0] for filename in os.listdir(directory)}

        def fix_escapes(entries: List[str]) -> List[str]:
            return [entry.replace("\\", "\\\\") for entry in entries]

        def build_entry(name: str, use_pitch_guidance: bool, gt_wavs_dir: str, feature_dir: str,
                        f0_dir: str, f0nsf_dir: str, speaker_index: int) -> str:
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

        if use_pitch_guidance:
            f_names = (
                    get_basenames(gt_wavs_dir) &
                    get_basenames(feature_dir) &
                    get_basenames(f0_dir) &
                    get_basenames(f0nsf_dir)
            )
        else:
            f_names = get_basenames(gt_wavs_dir) & get_basenames(feature_dir)

        opt = [build_entry(f_name, use_pitch_guidance, gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir, speaker_index)
               for f_name in f_names]

        fea_dim = 256 if model_version == "v1" else 768
        mutes_dir = os.path.join(app_path, "modules", "rvc")
        mute_wav_path = os.path.join(mutes_dir, "logs", "mute", "0_gt_wavs")
        mute_feature_path = os.path.join(mutes_dir, "logs", "mute", f"3_feature{fea_dim}")
        mute_f0_path = os.path.join(mutes_dir, "logs", "mute", "2a_f0")
        mute_f0nsf_path = os.path.join(mutes_dir, "logs", "mute", "2b-f0nsf")

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
                    config_obj.json_config[config_path],
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                f.write("\n")
        config_save_path = os.path.join(exp_dir, "config.json")
        with open(config_save_path, "r") as f:
            config_loaded = json.load(f)
        hparams = HParams(**config_loaded)
        hparams.model_dir = hparams.experiment_dir = exp_dir
        hparams.save_epoch_frequency = save_epoch_frequency
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
        hparams.save_latest_only = save_latest_only
        hparams.save_every_weights = 1 if save_weights_each_ckpt == "Yes" else 0
        hparams.if_cache_data_in_gpu = 1 if cache_dataset_to_gpu == "Yes" else 0
        hparams.data.training_files = os.path.join(exp_dir, "filelist.txt")

        logger.info(f"Training with hparams: {hparams}")
        train_main(hparams, progress)
        return "Training complete."
    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


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
            big_npy = MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * config.n_cpu,
                compute_labels=False,
                init="random",
            ).fit(big_npy).cluster_centers_
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
    trained_index_path = os.path.join(exp_dir,
                                      f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_version}.index")
    faiss.write_index(index, trained_index_path)
    infos.append("Adding data to the index...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])
    added_index_path = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{model_version}.index")
    faiss.write_index(index, added_index_path)
    infos.append(f"Successfully built index: {added_index_path}")
    try:
        logger.info(f"Copying index to {index_root}")
        shutil.copy(added_index_path, os.path.join(index_root, f"{os.path.basename(exp_dir)}.index"))
        infos.append(f"Linked index to external location: {outside_index_root}")
    except Exception:
        infos.append(f"Failed to link index to external location: {outside_index_root}")
    dataset_length = get_dataset_length(os.path.join(index_root, f"{os.path.basename(exp_dir)}.index"))
    infos.append(f"Dataset contains {dataset_length} total vectors")

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
        pause_after_separation=False,
        progress=gr.Progress()
):
    infos = []
    resuming_training = False
    if (not project_name or project_name == "") and (not existing_project_name or existing_project_name == ""):
        return ("Please provide a project name.", gr.update(visible=False))
    if project_name and existing_project_name:
        return ("Please provide only one project name.", gr.update(visible=False))
    if (not project_name or project_name == "") and existing_project_name:
        logger.info("Using existing project name")
        project_name = existing_project_name
        resuming_training = True
        input_dir = os.path.join(output_path, "voices", project_name, "raw")
        if not os.path.exists(input_dir):
            return (f"Project {project_name} does not exist. Please provide a valid project name.", 
                   gr.update(visible=False))
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        inputs = input_files
    if not inputs:
        return ("Please provide input files.", gr.update(visible=False))

    def get_info_str(strr):
        infos.append(strr)
        logger.info(strr)
        return "\n".join(infos)

    exp_dir = os.path.join(output_path, "voices", project_name)
    data_dir = os.path.join(output_path, "voices", project_name, "raw")
    os.makedirs(data_dir, exist_ok=True)
    if num_cpus == 0:
        num_cpus = 1

    # Analyze pitch range before processing
    yield (get_info_str("Step 1: Analyzing pitch range of input files..."), gr.update(visible=False))
    pitch_info = analyze_pitch_range(inputs, progress)
    if pitch_info:
        yield (get_info_str(f"Pitch Analysis Results:"), gr.update(visible=False))
        yield (get_info_str(f"Min Pitch: {pitch_info['min_pitch']:.2f} Hz"), gr.update(visible=False))
        yield (get_info_str(f"Max Pitch: {pitch_info['max_pitch']:.2f} Hz"), gr.update(visible=False))
        yield (get_info_str(f"Average Pitch: {pitch_info['avg_pitch']:.2f} Hz"), gr.update(visible=False))
        yield (get_info_str(f"Pitch Range: {pitch_info['pitch_range']:.2f} Hz"), gr.update(visible=False))

    # Continue with preprocessing and training
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if project_version == "v1" else "3_feature768")
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    missing_dirs = []
    for d in [gt_wavs_dir, feature_dir, f0_dir, f0nsf_dir, data_dir]:
        if not os.path.exists(d):
            missing_dirs.append(d)

    # Preprocess
    try:
        if not resuming_training or len(missing_dirs) > 0:
            yield (get_info_str("Step1: Preprocessing data."), gr.update(visible=False))
            vocal_files = inputs
            if separate_vocals:
                vocal_files, bg_vocal_files = separate_vocal(inputs, progress)
                yield (get_info_str(f"Separated vocals from {len(vocal_files)} files."), gr.update(visible=False))
            
            # Process each file
            for index, f in enumerate(vocal_files):
                progress(index / len(vocal_files), f"Processing {f} ({index + 1}/{len(vocal_files)})")
                try:
                    current_dir = os.path.dirname(f)
                    base_name, ext = os.path.splitext(os.path.basename(f))
                    output_file = os.path.join(data_dir, f"{base_name}.wav")
                    
                    # Skip if file already exists in raw folder and we're not in a fresh project
                    if os.path.exists(output_file) and not (not resuming_training and len(missing_dirs) > 0):
                        logger.info(f"Skipping existing file: {output_file}")
                        continue
                        
                    if os.path.exists(output_file):
                        f = output_file
                    audio, samplerate = torchaudio.load(f)
                    if audio.shape[0] == 2:
                        left_channel = audio[0].unsqueeze(0)
                        logger.info(f"Converting stereo to mono: {f}")
                        torchaudio.save(output_file, left_channel, sample_rate=samplerate)
                    else:
                        if not os.path.exists(output_file):
                            torchaudio.save(output_file, audio, sample_rate=samplerate)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error processing file {f}: {e}")
            
            if pause_after_separation:
                yield (get_info_str("Vocal separation complete. Click Resume to continue processing."), 
                       gr.update(visible=True))
                return (get_info_str("Vocal separation complete. Click Resume to continue processing."), 
                       gr.update(visible=True))
                
            preprocess_dataset(data_dir, exp_dir, tgt_sample_rate, num_cpus, progress)
        else:
            progress(0.25, "Skipping pitch feature extraction.")
            yield (get_info_str("Step1: Data already preprocessed."), gr.update(visible=False))
            yield (get_info_str("Step2: Pitch features already extracted."), gr.update(visible=False))

    except Exception as e:
        error_msg = f"Error during preprocessing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield (get_info_str(error_msg), gr.update(visible=False))
        return (get_info_str(error_msg), gr.update(visible=False))

    # Extract pitch features
    try:
        yield (get_info_str("Step2: Extracting pitch features."), gr.update(visible=False))
        progress(0.25, "Extracting pitch features.")
        extract_f0_feature(num_cpus, extraction_method, use_pitch_guidance, exp_dir, project_version, gpus_rmvpe)
    except Exception as e:
        error_msg = f"Error during pitch feature extraction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield (get_info_str(error_msg), gr.update(visible=False))
        return (get_info_str(error_msg), gr.update(visible=False))

    # First build the index
    yield (get_info_str("Step 2: Building initial index..."), gr.update(visible=False))
    [get_info_str(_) for _ in train_index(project_name, project_version)]

    # Training model
    try:
        progress(0.5, "Training model.")
        yield (get_info_str("Step3: Training model."), gr.update(visible=False))
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
    except Exception as e:
        error_msg = f"Error during model training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield (get_info_str(error_msg), gr.update(visible=False))
        return (get_info_str(error_msg), gr.update(visible=False))

    # Index building
    try:
        index_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}.index")
        if not os.path.exists(index_file):
            progress(0.75, "Building index")
            yield (get_info_str("Step4: Training complete, now building index."), gr.update(visible=False))
            [get_info_str(_) for _ in train_index(project_name, project_version)]
        else:
            yield (get_info_str("Step4: Index already exists."), gr.update(visible=False))
    except Exception as e:
        error_msg = f"Error during index building: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield (get_info_str(error_msg), gr.update(visible=False))
        return (get_info_str(error_msg), gr.update(visible=False))

    yield (get_info_str("Processing complete!"), gr.update(visible=False))
    return (get_info_str("Processing complete!"), gr.update(visible=False))


def resume_training(
        project_name,
        existing_project_name,
        tgt_sample_rate,
        use_pitch_guidance,
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
    if (not project_name or project_name == "") and (not existing_project_name or existing_project_name == ""):
        return ("Please provide a project name.", gr.update(visible=False))
    if project_name and existing_project_name:
        project_name = existing_project_name
    
    exp_dir = os.path.join(output_path, "voices", project_name)
    data_dir = os.path.join(output_path, "voices", project_name, "raw")
    
    if not os.path.exists(data_dir):
        return ("Project data directory not found. Please start training from the beginning.", gr.update(visible=False))
        
    for output in train1key(
        project_name,
        existing_project_name,
        False,  # separate_vocals
        tgt_sample_rate,
        use_pitch_guidance,
        None,  # inputs
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
        False,  # pause_after_separation
        progress
    ):
        yield output


def do_train_index(project_name, existing_project, project_version, progress=gr.Progress()):
    if existing_project is not None and existing_project != "":
        project_name = existing_project
    exp_dir = os.path.join(output_path, "voices", project_name)
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        logger.info(strr)
        return "\n".join(infos)

    try:
        if not os.path.exists(exp_dir):
            yield get_info_str("Project not found.")
            return
        model_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}_final.pth")
        model_file_new = os.path.join(index_root, f"{os.path.basename(exp_dir)}.pth")
        if os.path.exists(model_file_new):
            model_file = model_file_new
        if not os.path.exists(model_file):
            yield get_info_str("Model file not found.")
            return
        with open(model_file, "rb") as f:
            model_dict = torch.load(f)
            vers = model_dict.get("version", "v2")
        index_file = os.path.join(index_root, f"{os.path.basename(exp_dir)}.index")
        progress(0.25, "Building index")
        yield get_info_str("Step4: Training complete, now building index.")
        [get_info_str(_) for _ in train_index(project_name, project_version)]
        progress(1, "Processing complete.")
        yield get_info_str("Processing complete!")
    except Exception as e:
        error_msg = f"Error during index training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield error_msg


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
        gr.Markdown("# 🎤 Voice Model Training")
        gr.Markdown("Train voice conversion models with 30-60 minutes of audio. Features automatic vocal separation, feature extraction, and customizable training parameters for creating personalized voice models.")
        with gr.Row():
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
                    maximum=4000,
                    step=5,
                    label="Total Training Epochs",
                    value=300,
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
                pause_after_separation = gr.Checkbox(
                    label="Pause After Separation",
                    value=False,
                    elem_classes="hintitem", elem_id="rvc_pause_after_separation"
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
                        choices=["hybrid", "pm", "harvest", "dio", "rmvpe", "rmvpe_onnx", "rmvpe+", "crepe",
                                 "crepe-tiny",
                                 "mangio-crepe", "mangio-crepe-tiny"],
                        value="rmvpe+",
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
                        choices=["Yes", "No"],
                        value="Yes",
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
                    fn=change_version19,
                    inputs=[sample_rate, use_pitch_guidance, model_version],
                    outputs=[pretrained_generator, pretrained_discriminator, sample_rate],
                )
                use_pitch_guidance.change(
                    fn=change_f0,
                    inputs=[use_pitch_guidance, sample_rate, model_version],
                    outputs=[pitch_extraction_method, gpus_rmvpe, pretrained_generator, pretrained_discriminator],
                )
            with gr.Column():
                gr.Markdown("### 🎤 Inputs")
                input_files = gr.File(
                    label="Audio Files",
                    file_count="multiple",
                    file_types=["audio", "video"],
                    elem_classes="hintitem", elem_id="rvc_input_files"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        input_url = gr.Textbox(
                            label="Audio URLs",
                            placeholder="Enter URLs separated by a new line",
                            elem_classes="hintitem", elem_id="rvc_input_url"
                        )
                    with gr.Column():
                        input_url_button = gr.Button(
                            value='Load URLs',
                            variant='secondary',
                            elem_classes="hintitem", elem_id="rvc_input_url_button"
                        )
            
            with gr.Column():
                gr.Markdown("### 🎮 Actions")
                with gr.Row():
                    start_train = gr.Button(
                        "Start Training",
                        elem_classes="hintitem", elem_id="rvc_start_train", variant="primary",
                    )
                    resume_train = gr.Button(
                        "Resume",
                        elem_classes="hintitem", elem_id="rvc_resume_train", variant="primary",
                        visible=False
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
                
                gr.Markdown("### 🎶 Outputs")
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
                        gpus_rmvpe,
                        pause_after_separation
                    ],
                    [info3, resume_train],
                    api_name="train_start_all",
                )
                resume_train.click(
                    resume_training,
                    [
                        voice_name,
                        existing_project,
                        sample_rate,
                        use_pitch_guidance,
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
                    [info3, resume_train],
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
        "pause_after_separation": "Check this box to pause processing after vocal separation is complete. Useful if you need to further refine the vocal tracks before training.",
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
        "resume_train": "Click to continue processing after vocal separation is complete.",
        "cancel_train": "Click to cancel the training process if needed.",
        "output_info": "Displays logs and training progress information.",
        "train_index": "Click to build or re-build the index for the trained voice model.",
    }
    for elem_id, description in descriptions.items():
        arg_handler.register_description("rvc", elem_id, description)


def register_api_endpoints(api):
    """
    Register API endpoints for RVC voice training
    
    Args:
        api: FastAPI application instance
    """
    from fastapi import UploadFile, File, Form, BackgroundTasks, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from typing import Optional, List
    
    @api.post("/api/v1/rvc/train", tags=["RVC"])
    async def api_train_rvc_model(
        background_tasks: BackgroundTasks,
        project_name: str = Form(...),
        sample_rate: int = Form(48000),
        use_pitch_guidance: bool = Form(True),
        speaker_id: int = Form(0),
        extraction_method: str = Form("rmvpe+"),
        epoch_save_freq: int = Form(10),
        train_epochs: int = Form(200),
        batch_size: int = Form(16),
        save_latest_only: bool = Form(False),
        save_weights_every: bool = Form(True),
        project_version: str = Form("v2"),
        gpus_rmvpe: str = Form("0"),
        audio_files: List[UploadFile] = File(...)
    ):
        """
        Train a new RVC voice model
        
        Args:
            background_tasks: FastAPI background tasks
            project_name: Name for the voice project
            sample_rate: Target sample rate for the model
            use_pitch_guidance: Whether to use pitch guidance
            speaker_id: Speaker ID for the model
            extraction_method: Method for extracting features
            epoch_save_freq: How often to save checkpoints
            train_epochs: Total number of training epochs
            batch_size: Training batch size
            save_latest_only: Whether to save only the latest checkpoint
            save_weights_every: Whether to save weights with each checkpoint
            project_version: RVC model version
            gpus_rmvpe: GPU indices for RMVPE
            audio_files: Audio files for training
            
        Returns:
            Status information and job ID
        """
        try:
            # Validate inputs
            if not project_name or project_name.strip() == "":
                raise HTTPException(status_code=400, detail="Project name cannot be empty")
                
            if not audio_files or len(audio_files) == 0:
                raise HTTPException(status_code=400, detail="No audio files provided")
                
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix="rvc_train_")
            
            # Save uploaded audio files
            audio_paths = []
            for audio_file in audio_files:
                file_path = os.path.join(temp_dir, audio_file.filename)
                with open(file_path, "wb") as f:
                    content = await audio_file.read()
                    f.write(content)
                audio_paths.append(file_path)
            
            # Start training in the background
            job_id = f"train_{int(time.time())}"
            background_tasks.add_task(
                run_train_job,
                job_id=job_id,
                project_name=project_name,
                audio_files=audio_paths,
                sample_rate=sample_rate,
                use_pitch_guidance=use_pitch_guidance,
                speaker_id=speaker_id,
                extraction_method=extraction_method,
                epoch_save_freq=epoch_save_freq,
                train_epochs=train_epochs,
                batch_size=batch_size,
                save_latest_only=save_latest_only,
                save_weights_every=save_weights_every,
                project_version=project_version,
                gpus_rmvpe=gpus_rmvpe,
                temp_dir=temp_dir
            )
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": f"Training job started for project '{project_name}' with {len(audio_files)} audio files",
                "estimated_time": f"Estimated time: {len(audio_files) * 2 + train_epochs * 0.5:.1f} minutes"
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception("Error starting RVC training:")
            raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
            
    @api.get("/api/v1/rvc/models", tags=["RVC"])
    async def api_list_rvc_models():
        """
        List available RVC voice models
        
        Returns:
            List of available voice models
        """
        try:
            voice_projects = list_voice_projects()
            
            result = {
                "models": []
            }
            
            for project in voice_projects:
                # Get the weights for this project
                project_dir = os.path.join(output_path, "voices", project)
                weights = list_project_weights(project_dir)
                
                # Get project version (v1 or v2)
                version_file = os.path.join(project_dir, "version.txt")
                version = "v2"  # Default
                if os.path.exists(version_file):
                    with open(version_file, "r") as f:
                        version = f.read().strip()
                
                result["models"].append({
                    "name": project,
                    "version": version,
                    "weights": weights
                })
                
            return result
            
        except Exception as e:
            logger.exception("Error listing RVC models:")
            raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
            
    @api.get("/api/v1/rvc/job/{job_id}", tags=["RVC"])
    async def api_get_job_status(job_id: str):
        """
        Get status of a training job
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Current status of the job
        """
        try:
            status_dir = os.path.join(output_path, "voices", "jobs")
            status_file = os.path.join(status_dir, f"{job_id}.json")
            
            if not os.path.exists(status_file):
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
                
            with open(status_file, "r") as f:
                status = json.load(f)
                
            return status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error getting job status for {job_id}:")
            raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")
            
    @api.get("/api/v1/rvc/download/{project_name}/{weight_file}", tags=["RVC"])
    async def api_download_model(project_name: str, weight_file: str):
        """
        Download a trained model weight file
        
        Args:
            project_name: Name of the voice project
            weight_file: Name of the weight file to download
            
        Returns:
            Model weight file
        """
        try:
            project_dir = os.path.join(output_path, "voices", project_name)
            if not os.path.exists(project_dir):
                raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
                
            weight_path = os.path.join(project_dir, weight_file)
            if not os.path.exists(weight_path):
                raise HTTPException(status_code=404, detail=f"Weight file {weight_file} not found")
                
            return FileResponse(
                weight_path,
                media_type="application/octet-stream",
                filename=weight_file
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error downloading model {project_name}/{weight_file}:")
            raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

    @api.post("/api/v1/rvc/upload", tags=["RVC"])
    async def api_upload_datasets(
        project_name: str = Form(...),
        dataset_files: List[UploadFile] = File(...)
    ):
        # Implementation of the new API endpoint
        pass

def run_train_job(
    job_id, project_name, audio_files, sample_rate, use_pitch_guidance, 
    speaker_id, extraction_method, epoch_save_freq, train_epochs,
    batch_size, save_latest_only, save_weights_every, project_version, 
    gpus_rmvpe, temp_dir
):
    """Run a training job in the background"""
    try:
        # Create status directory
        status_dir = os.path.join(output_path, "voices", "jobs")
        os.makedirs(status_dir, exist_ok=True)
        
        # Initialize status
        status = {
            "job_id": job_id,
            "project_name": project_name,
            "status": "preprocessing",
            "progress": 0,
            "start_time": time.time(),
            "completion_time": None,
            "current_epoch": 0,
            "total_epochs": train_epochs,
            "log": []
        }
        
        # Function to update and save status
        def update_status(new_status, progress=None, message=None):
            nonlocal status
            status["status"] = new_status
            if progress is not None:
                status["progress"] = progress
            if message:
                status["log"].append({
                    "time": time.time(),
                    "message": message
                })
            # Save to file
            with open(os.path.join(status_dir, f"{job_id}.json"), "w") as f:
                json.dump(status, f, indent=2)
        
        # Start training
        update_status("preprocessing", 0, "Starting preprocessing")
        
        # Get number of CPUs
        num_cpus = os.cpu_count() or 4
        
        # Run the actual training process
        train1key(
            project_name=project_name,
            existing_project_name=None,
            separate_vocals=True,
            tgt_sample_rate=sample_rate,
            use_pitch_guidance=use_pitch_guidance,
            inputs=audio_files,
            spk_id=speaker_id,
            num_cpus=num_cpus,
            extraction_method=extraction_method,
            epoch_save_freq=epoch_save_freq,
            train_epochs=train_epochs,
            batch_size=batch_size,
            save_latest=save_latest_only,
            generator="",
            discriminator="",
            tgt_gpus="0",
            cache_to_gpu=False,
            save_weights_every=save_weights_every,
            project_version=project_version,
            gpus_rmvpe=gpus_rmvpe,
            pause_after_separation=False,
            progress=StatusUpdater(update_status)
        )
        
        # Update final status
        update_status("completed", 100, "Training completed successfully")
        status["completion_time"] = time.time()
        with open(os.path.join(status_dir, f"{job_id}.json"), "w") as f:
            json.dump(status, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error in training job {job_id}: {e}")
        
        # Update failed status
        try:
            status["status"] = "failed"
            status["completion_time"] = time.time()
            status["log"].append({
                "time": time.time(),
                "message": f"Error: {str(e)}"
            })
            with open(os.path.join(status_dir, f"{job_id}.json"), "w") as f:
                json.dump(status, f, indent=2)
        except Exception as status_e:
            logger.error(f"Error updating job status: {status_e}")
            
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")

class StatusUpdater:
    """
    Helper class to update job status from progress callback
    """
    def __init__(self, update_func):
        self.update_func = update_func
        self.last_update = 0
        
    def __call__(self, value, desc="", total=100):
        # Calculate percentage and only update status occasionally to avoid too many files writes
        percentage = int((value / total) * 100) if total > 0 else 0
        
        # Update if significant change (more than 5%) or if it's at the beginning or end
        if (percentage - self.last_update >= 5) or percentage == 0 or percentage == 100:
            self.last_update = percentage
            
            # Determine status based on description
            if "preprocess" in desc.lower():
                status = "preprocessing"
            elif "extract" in desc.lower():
                status = "feature_extraction"
            elif "train" in desc.lower():
                status = "training"
                # Extract current epoch if available
                if "epoch" in desc.lower():
                    try:
                        epoch_parts = desc.split("epoch")
                        if len(epoch_parts) > 1:
                            epoch_part = epoch_parts[1].strip()
                            current_epoch = int(epoch_part.split("/")[0])
                            self.update_func(status, percentage, desc)
                            return
                    except Exception:
                        pass
            else:
                status = "processing"
                
            self.update_func(status, percentage, desc)
