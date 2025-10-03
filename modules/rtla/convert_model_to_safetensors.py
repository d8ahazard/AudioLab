import json
import os
import sys
import torch
from safetensors.torch import save_file as safe_save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from modules.rtla.config import MODELS_DIR, CRNN_MODEL_PT, CRNN_MODEL_SAFE, CRNN_CONFIG_JSON


def main():
    if not os.path.exists(CRNN_MODEL_PT):
        print(f".pt model not found at {CRNN_MODEL_PT}")
        sys.exit(1)

    save_data = torch.load(CRNN_MODEL_PT, map_location="cpu")
    state = save_data["model_state_dict"]
    config = save_data["config"]
    consts = save_data["consts"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_save_file(state, CRNN_MODEL_SAFE)
    with open(CRNN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump({"config": config, "consts": consts}, f, indent=2)

    print(f"Saved {CRNN_MODEL_SAFE} and {CRNN_CONFIG_JSON}")


if __name__ == "__main__":
    main()


