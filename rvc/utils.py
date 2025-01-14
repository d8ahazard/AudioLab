import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import torch

from handlers.config import model_path

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # 模型需要的shape
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning(
                        "shape-%s-mismatch. need: %s, get: %s",
                        k,
                        state_dict[k].shape,
                        saved_state_dict[k].shape,
                    )  #
                    raise KeyError
            except:
                # logger.info(traceback.format_exc())
                logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
                new_state_dict[k] = v  # 模型自带的随机值
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
            optimizer is not None and load_opt == 1
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    else:
        iteration = 1
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )  #
                raise KeyError
        except:
            # logger.info(traceback.format_exc())
            logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    else:
        iteration = 1
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, accelerator):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    state_dict = accelerator.get_state_dict(model)
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    torch.save(
        {
            "combd": state_dict_combd,
            "sbd": state_dict_sbd,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_dir", type=str, required=True, help="experiment dir"
    )
    parser.add_argument(
        "--resume", type=str, required=False, default=None, help="resume accelerate state path"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="save dir (overwrites assets/weights/)"
    )
    parser.add_argument(
        "--config", type=str, help="config", default="configs/v3/40k.json"
    )
    parser.add_argument(
        "--data_root", type=Path, help="data root prefix for filelist", default=None
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        required=False, default=5,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "--total_epoch", type=int, required=True, help="total_epoch"
    )
    parser.add_argument(
        "--pretrainG", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument(
        "--pretrainD", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument(
        "--enable_opt_load", type=int, default=1, help="enable optimizer load from checkpoint on finetuning"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="batch size"
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay", type=float, required=True, help="lr decay each epoch"
    )
    parser.add_argument(
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "--log_every", type=int, help="wandb log interval", default=200
    )
    parser.add_argument(
        "--if_ppg",
        type=int,
        default=0,
        help="whether to use ppg features as an additional input, 1 or 0",
    )
    parser.add_argument(
        "--enable_perturbation",
        type=int,
        default=0,
        help="whether to use ppg/vae perturbation during training, 1 or 0",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="coveroke",
    )
    parser.add_argument(
        "--if_latest",
        type=int,
        default=0,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "--if_cache_data_in_gpu",
        type=int,
        required=False, default=0,
        help="if caching the dataset in GPU memory, 1 or 0",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=10000000,
    )

    args = parser.parse_args()
    # name = args.experiment_dir
    # experiment_dir = os.path.join("./logs", args.experiment_dir)
    experiment_dir = args.exp_dir

    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)

    hparams.resume = args.resume
    hparams.data.training_files = f"{experiment_dir}/filelist.txt"
    hparams.data.data_root = args.data_root

    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_dir = args.save_dir
    hparams.save_interval = args.save_interval
    hparams.name = Path(experiment_dir).name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.enable_opt_load = args.enable_opt_load
    hparams.train.batch_size = args.batch_size
    hparams.train.total_steps = args.total_steps
    hparams.if_ppg = args.if_ppg
    hparams.enable_perturbation = args.enable_perturbation
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu

    hparams.train.learning_rate = args.lr
    hparams.train.lr_decay = args.lr_decay
    hparams.train.log_interval = args.log_every

    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warning(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


# class HParams:
#     """
#     Updated HParams class that creates default nested sections (train, data, model)
#     and then overrides them with any provided kwargs.
#
#     The code below preserves the original HParams "dictionary-like" behavior:
#     you can still do hps["foo"] and hps.foo interchangeably.
#     """
#
#     def __init__(self, **kwargs):
#         # 1. Define the entire default structure as a nested dictionary
#         default_dict = {
#             ################# Top-level keys #################
#             "model_dir": os.path.join(model_path, "rvc"),
#             "pretrainG": "",
#             "pretrainD": "",
#             "save_interval": 5,
#             "enable_opt_load": 1,  # if you want to load optimizer from checkpoint
#             "if_ppg": 0,
#             "if_latest": 0,
#             "save_every_weights": "0",
#             "enable_perturbation": 0,
#             "version": "v1",
#             "name": "unnamed_experiment",
#             "sample_rate": 40000,  # default sample rate
#
#             ################# Nested sections #################
#             "train": {
#                 "seed": 1234,
#                 "epochs": 100,  # how many total epochs to train
#                 "batch_size": 2,
#                 "learning_rate": 1e-4,
#                 "lr_decay": 0.999,
#                 "segment_size": 8192,  # number of audio samples per training segment
#                 "log_interval": 200,
#                 "total_steps": 999999999,  # large number if you want to rely on epochs
#                 "betas": [0.8, 0.99],
#                 "eps": 1e-9,
#                 "c_mel": 45,  # used in RVC's code for mel loss weight
#                 "c_kl": 1.0,  # used in RVC's code for kl loss weight
#             },
#
#             "data": {
#                 "training_files": "filelist.txt",  # could be changed to your real training list
#                 "filter_length": 1024,
#                 "hop_length": 256,
#                 "win_length": 1024,
#                 "sampling_rate": 40000,
#                 "n_mel_channels": 80,
#                 "mel_fmin": 0.0,
#                 "mel_fmax": None,
#                 "data_root": None,
#             },
#
#             "model": {
#                 "use_spectral_norm": False,
#                 "inter_channels": 192,
#                 "hidden_channels": 192,
#                 "filter_channels": 768,
#                 "n_heads": 2,
#                 "n_layers": 6,
#                 "kernel_size": 3,
#                 "p_dropout": 0,
#                 "resblock": "1",
#                 "resblock_kernel_sizes": [3, 7, 11],
#                 "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
#                 "upsample_rates": [10, 10, 2, 2],
#                 "upsample_initial_channel": 512,
#                 "upsample_kernel_sizes": [16, 16, 4, 4],
#                 "spk_embed_dim": 109,
#                 "gin_channels": 256,
#             }
#         }
#
#         # 2. Recursively update this default_dict with any user-provided kwargs
#         #    This ensures that nested dictionaries get updated properly.
#         def nested_update(d, u):
#             for k, v in u.items():
#                 if isinstance(v, dict) and k in d and isinstance(d[k], dict):
#                     nested_update(d[k], v)
#                 else:
#                     d[k] = v
#
#         nested_update(default_dict, kwargs)
#
#         # 3. Now, attach everything to self in a nested manner
#         #    so we end up with self.train, self.data, self.model, etc.
#         for k, v in default_dict.items():
#             if isinstance(v, dict):
#                 # If a value is itself a dict, convert it into an HParams instance for chain-access
#                 setattr(self, k, HParams(**v))
#             else:
#                 setattr(self, k, v)
#
#     ########################
#     # The usual HParams dictionary-like methods
#     ########################
#     def keys(self):
#         return self.__dict__.keys()
#
#     def items(self):
#         return self.__dict__.items()
#
#     def values(self):
#         return self.__dict__.values()
#
#     def __len__(self):
#         return len(self.__dict__)
#
#     def __getitem__(self, key):
#         return getattr(self, key)
#
#     def __setitem__(self, key, value):
#         return setattr(self, key, value)
#
#     def __contains__(self, key):
#         return key in self.__dict__
#
#     def __repr__(self):
#         return self.__dict__.__repr__()

def savee(ckpt, sr, name, epoch, version, hps):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["version"] = version
        save_path = str(Path(hps.save_dir) / f"{name}.pth")
        torch.save(opt, save_path)
        print(f"Saved model to {save_path}")
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path):
    try:
        a = torch.load(path, map_location="cpu")
        return "模型信息:%s\n采样率:%s\n模型是否输入音高引导:%s\n版本:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except:
        return traceback.format_exc()


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, "assets/weights/%s" % name)
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:

        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                        alpha1 * (ckpt1[key][:min_shape0].float())
                        + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                        alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["sr"] = sr
        opt["f0"] = 1 if f0 else 0
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, "assets/weights/%s.pth" % name)
        return "Success."
    except:
        return traceback.format_exc()
