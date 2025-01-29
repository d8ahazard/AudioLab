import os
import traceback

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from handlers.config import model_path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def extract_feature_print(device, exp_dir, version, is_half):
    if "privateuseone" not in device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    else:
        import torch_directml

        device = torch_directml.device(torch_directml.default_device())

        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res

        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

    f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

    model_file = os.path.join(model_path, "rvc", "hubert_base.pt")

    print("exp_dir: " + exp_dir)
    wavPath = "%s/1_16k_wavs" % exp_dir
    outPath = (
        "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
    )
    os.makedirs(outPath, exist_ok=True)

    # HuBERT model
    print("load model(s) from {}".format(model_file))
    # if hubert model is exist
    if not os.access(model_path, os.F_OK):
        print(
            "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            % model_path
        )
        exit(0)
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_file],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    print("move model to %s" % device)
    if is_half:
        if device not in ["mps", "cpu"]:
            model = model.half()
    model.eval()

    todo = sorted(list(os.listdir(wavPath)))
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        print("no-feature-todo")
    else:
        print("all-feature-%s" % len(todo))
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if is_half and device not in ["mps", "cpu"]
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        print("%s-contains nan" % file)
                    if idx % n == 0:
                        print("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
            except:
                print(traceback.format_exc())
        print("all-feature-done")
