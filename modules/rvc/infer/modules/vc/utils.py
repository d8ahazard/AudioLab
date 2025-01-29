import os

from fairseq import checkpoint_utils

from handlers.config import model_path


def get_index_path_from_model(sid):
    index_file = sid.replace("_final.pth", "_index.index")
    if os.path.exists(index_file):
        return index_file
    return ""


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
