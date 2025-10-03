import os
import numpy as np
from handlers.config import model_path as _BASE_MODEL_PATH

# Use AudioLab's config handler for model root
MODELS_DIR = os.path.join(_BASE_MODEL_PATH, "rtla")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_PIECE_ID = 12
CHANNELS = 1
SAMPLE_RATE = 16000
HOP_LENGTH = 640  # 40ms
N_FFT = 2 * HOP_LENGTH
N_MELS = 66
NORM = np.inf
CHUNK_SIZE = 4 * HOP_LENGTH
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH
DTW_WINDOW_SIZE = int(3 * FRAME_RATE)

CRNN_MODEL_PT = os.path.join(MODELS_DIR, "pretrained-model.pt")
CRNN_MODEL_SAFE = os.path.join(MODELS_DIR, "pretrained-model.safetensors")
CRNN_CONFIG_JSON = os.path.join(MODELS_DIR, "pretrained-model.json")

# Prefer safetensors if present
CRNN_MODEL_PATH = CRNN_MODEL_SAFE if os.path.exists(CRNN_MODEL_SAFE) else CRNN_MODEL_PT

# Default features to use for alignment
FEATURES = ["chroma", "phoneme"]

# Evaluation tolerances (ms) – used for internal diagnostics if needed
TOLERANCES = [200, 300, 500, 750, 1000]


