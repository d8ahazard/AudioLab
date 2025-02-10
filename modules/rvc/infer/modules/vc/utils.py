import os

from fairseq import checkpoint_utils

from handlers.config import model_path


def get_index_path_from_model(sid):
    sid_base = os.path.basename(sid)
    sid_base = sid_base.replace(".pth", "")
    sid_base = sid_base.replace("_final", "")
    # Find file that contains sid_base and .index
    for file in os.listdir(os.path.join(model_path, "trained")):
        if sid_base in file and ".index" in file:
            return file
    return None


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, "rvc", "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
