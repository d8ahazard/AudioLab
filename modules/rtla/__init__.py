"""Real-time lyrics alignment (RTLA) module integrated into AudioLab.

Provides:
- OLTW: Online/Offline Local Time Warping alignment to compute a warping path
- StreamProcessor: offline feature chunker (mock stream) for files
- utils: feature extraction (chroma, mel, phoneme), CRNN loader, path helpers
- config: constants and model path resolved within AudioLab tree
"""


