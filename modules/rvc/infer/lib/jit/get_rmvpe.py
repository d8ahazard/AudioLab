import torch
import os

from handlers.rvc_trainer import model_dir


def get_rmvpe(model_path=os.path.join(model_dir, "rvc", "rvmpe.pt"), device=torch.device("cpu")):
    from modules.rvc.infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
