import os
import traceback
import fairseq
import logging
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from handlers.config import model_path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

logger = logging.getLogger(__name__)


# wave must be 16k, hop_size=320
def read_wave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000, f"Expected sample rate 16000, got {sr}"
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # dual channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, f"Expected 1D features, got {feats.dim()}D"
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

    # Build paths using os.path.join
    log_file = os.path.join(exp_dir, "extract_f0_feature.log")
    # (Removed unused file handle for logging)

    model_file = os.path.join(model_path, "rvc", "hubert_base.pt")
    wav_dir = os.path.join(exp_dir, "1_16k_wavs")
    if version == "v1":
        features_dir = os.path.join(exp_dir, "3_feature256")
    else:
        features_dir = os.path.join(exp_dir, "3_feature768")
    os.makedirs(features_dir, exist_ok=True)

    logger.info("exp_dir: %s", exp_dir)
    logger.info("Load model(s) from %s", model_file)

    # Check for required files/directories
    if not os.path.exists(model_file):
        logger.error(
            "Error: Model file %s does not exist. Please download it from "
            "https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main", model_file
        )
        raise FileNotFoundError("Model file not found: " + model_file)
    if not os.path.exists(wav_dir):
        logger.error("Error: WAV directory %s does not exist.", wav_dir)
        raise FileNotFoundError("WAV directory not found: " + wav_dir)

    models, saved_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_file],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    logger.info("Move model to %s", device)
    if is_half:
        if device not in ["mps", "cpu"]:
            model = model.half()
    model.eval()

    wav_files = sorted(os.listdir(wav_dir))
    n = max(1, len(wav_files) // 10)  # Print at most 10 logs
    if len(wav_files) == 0:
        logger.info("No files to process in %s", wav_dir)
    else:
        logger.info("Total files to process: %s", len(wav_files))
        for idx, wav_file in enumerate(wav_files):
            try:
                if wav_file.endswith(".wav"):
                    input_wav_path = os.path.join(wav_dir, wav_file)
                    output_feature_path = os.path.join(features_dir, wav_file.replace("wav", "npy"))

                    if os.path.exists(output_feature_path):
                        continue

                    feats = read_wave(input_wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if is_half and device not in ["mps", "cpu"]
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9 or 12
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(output_feature_path, feats, allow_pickle=False)
                    else:
                        logger.warning("%s contains NaN values", wav_file)
                    if idx % n == 0:
                        logger.info("Progress: Total %s, Processed %s, File %s, Feature shape %s",
                                    len(wav_files), idx, wav_file, feats.shape)
            except Exception:
                logger.error(traceback.format_exc())
        logger.info("Feature extraction done for all files")
