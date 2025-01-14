from transformers import HubertModel, HubertConfig


def load_hubert(hubert_path="assets/hubert/hubert_base.pt", device='cpu', is_half=False):
    # Load the Hubert configuration and model weights
    config = HubertConfig.from_pretrained(hubert_path)
    hubert_model = HubertModel.from_pretrained(hubert_path, config=config)

    # Move the model to the desired device
    hubert_model = hubert_model.to(device)

    # Adjust precision if necessary
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()

    return hubert_model.eval()
