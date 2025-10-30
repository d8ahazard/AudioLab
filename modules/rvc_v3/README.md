# RVC V3: Next-Generation Retrieval-Based Voice Conversion

## Overview

RVC V3 is a comprehensive upgrade to RVC v2, introducing text/lyric conditioning, enhanced audio fidelity, and improved robustness to audio effects.

### Key Features

- **Text/Lyric Conditioning**: Explicit control over lyrics with style tags for precise vocal styling
- **Dual Content Encoders**: Combined HuBERT + Whisper for robust content features
- **High-Fidelity Audio**: Support for 44.1+ kHz sample rates with stereo output
- **Cross-Attention Architecture**: Generator with cross-attention to text features
- **Style Tags**: Control vocal characteristics with tags like `[clean]`, `[raspy]`, `[breathy]`
- **Enhanced Retrieval**: FAISS-based feature retrieval with advanced mixing strategies

## Project Structure

```
modules/rvc_v3/
├── data_prep/          # Data preparation tools
│   ├── song_downloader.py    # Download audio from URLs
│   ├── vocal_separator.py     # Vocal isolation
│   ├── transcriber.py          # Automatic transcription
│   ├── lyric_editor.py        # Lyric editing & tagging
│   └── phonemizer.py          # Text-to-phoneme conversion
├── models/              # Model architectures
│   ├── content_encoders.py    # HuBERT + Whisper encoders
│   ├── text_encoder.py        # Transformer text encoder
│   ├── generator.py           # V3 generator with cross-attention
│   ├── vocoder.py            # Stereo vocoder support
│   └── retrieval.py          # FAISS retrieval index
├── training/            # Training pipeline
│   ├── dataset.py            # V3 dataset class
│   ├── extract_features.py   # Feature extraction
│   ├── build_index.py        # Index building
│   └── train.py              # Training loop
├── inference/           # Inference pipeline
│   └── pipeline.py           # Complete inference pipeline
└── configs/             # Configuration
    └── v3_config.py          # V3 configuration system
```

## Quick Start

### 1. Data Preparation

```python
from modules.rvc_v3.data_prep import SongDownloader, VocalSeparator, Transcriber

# Download a song
downloader = SongDownloader("outputs/rvc_v3_data")
audio_path, metadata = downloader.download(
    "https://youtube.com/watch?v=...",
    "my_project"
)

# Separate vocals
separator = VocalSeparator("outputs/rvc_v3_data")
vocals_path, _ = separator.separate_vocals(audio_path, "my_project")

# Transcribe lyrics
transcriber = Transcriber("outputs/rvc_v3_data")
segments = transcriber.transcribe(vocals_path, "my_project")
```

### 2. Annotate Lyrics with Tags

```python
from modules.rvc_v3.data_prep import LyricEditor, LyricSegment

editor = LyricEditor("my_project", "outputs/rvc_v3_data")
editor.load_from_transcript({"segments": segments})

# Add style tags to segments
editor.apply_tag_to_range(0, 10, "clean")  # Clean vocals for first 10 segments
editor.apply_tag_to_range(11, 20, "raspy")  # Raspy for next segments

# Save annotated lyrics
editor.save("annotated_lyrics.json")
```

### 3. Feature Extraction

```python
from modules.rvc_v3.training import FeatureExtractor
from modules.rvc_v3.configs import get_default_config

config = get_default_config(sample_rate=44100)
extractor = FeatureExtractor(config, device="cuda", use_dual_encoder=True)

# Extract features for all audio files
extractor.extract_project("outputs/rvc_v3_data/my_project")
```

### 4. Build Retrieval Index

```python
from modules.rvc_v3.training import IndexBuilder

builder = IndexBuilder(
    feature_dim=config.get_content_feature_dim(),
    index_type="IVF",
    use_gpu=True
)

index_path = builder.build_from_project(
    "outputs/rvc_v3_data/my_project",
    config
)
```

### 5. Training

```python
from modules.rvc_v3.training import RVCV3Trainer
from modules.rvc_v3.data_prep import Phonemizer

trainer = RVCV3Trainer(
    config,
    "outputs/rvc_v3_data/my_project",
    device="cuda"
)

phonemizer = Phonemizer(language='en-us')
trainer.train(num_epochs=20, phonemizer=phonemizer)
```

### 6. Inference

```python
from modules.rvc_v3.inference import RVCV3Pipeline

# Initialize pipeline
pipeline = RVCV3Pipeline(
    model_path_or_checkpoint="checkpoints/checkpoint_10000.pt",
    config=config,
    device="cuda"
)

# Load retrieval index
pipeline.load_retrieval_index(index_path)

# Convert audio
output_audio, sr = pipeline.convert(
    audio_path="input.wav",
    lyrics="[clean] Never gonna give you up, [raspy] never gonna let you down",
    output_path="output.wav",
    index_rate=0.75,
    pitch_shift=0
)
```

## Configuration

RVC V3 uses a comprehensive configuration system:

```python
from modules.rvc_v3.configs import RVCV3Config, get_default_config

# Get default config for 44.1kHz
config = get_default_config(44100)

# Customize
config.use_dual_encoder = True
config.text_encoder_dim = 256
config.n_cross_attn_layers = 2
config.stereo_mode = "mono"  # or "shared", "dual"

# Save config
config.save("config.json")

# Load config
config = RVCV3Config.load("config.json")
```

## Style Tags

Style tags are inserted in square brackets within lyrics to control vocal characteristics:

### Common Style Tags

- `[clean]` - Clean, clear vocals
- `[raspy]` - Raspy, rough texture
- `[breathy]` - Breathy, soft delivery
- `[distorted]` - Distorted, harsh vocals
- `[nasal]` - Nasal quality
- `[soft]` - Soft, gentle vocals
- `[belted]` - Powerful, belted vocals
- `[whisper]` - Whispered vocals
- `[growl]` - Growling vocals
- `[vibrato]` - With vibrato
- `[straight]` - Straight tone (no vibrato)
- `[head_voice]` - Head voice register
- `[chest_voice]` - Chest voice register
- `[falsetto]` - Falsetto register

### Custom Tags

You can create custom tags by annotating your training data:

```python
editor.add_segment(LyricSegment(
    start=10.0,
    end=15.0,
    text="Amazing vocals",
    tags=["my_custom_style"]
))
```

The model will learn to associate these tags with the vocal characteristics present in those segments.

## Training Tips

1. **Data Quality**: Use high-quality, dry vocals (no reverb, minimal background noise)
2. **Data Quantity**: 10-60 minutes of clean vocals recommended
3. **Annotation**: More tagged segments = better style control
4. **Sample Rate**: Higher is better (44.1kHz or 48kHz recommended)
5. **Dual Encoder**: Use dual encoder (HuBERT + Whisper) for best robustness
6. **Retrieval Rate**: Start with 0.75, adjust based on results (higher = more source, lower = more target)

## Requirements

### Core Dependencies
- torch >= 2.0.0
- fairseq
- whisper (or faster-whisper)
- faiss-gpu
- librosa
- soundfile
- gradio

### Optional Dependencies
- whisperx (for better word alignment)
- phonemizer (or use built-in espeak-ng)
- yt-dlp (for downloading)

## Troubleshooting

### Issue: "FAISS index not found"
- Make sure to build the index after feature extraction
- Check that feature files exist in `{project}/features/`

### Issue: "Out of memory during training"
- Reduce batch size
- Use FP16 training
- Use smaller model (reduce hidden dimensions)

### Issue: "Poor lyric intelligibility"
- Ensure lyrics are correctly transcribed and aligned
- Increase text encoder capacity
- Add more cross-attention layers

### Issue: "Style tags not working"
- Make sure training data has tagged segments
- Tags must be consistent (same spelling/capitalization)
- More tagged examples = better learning

## Architecture Details

### Content Encoding
- HuBERT base (768-D) for acoustic features
- Whisper (1280-D) for robust phonetic features
- Fusion via concatenation, addition, or learned mixing

### Text Encoding
- Transformer encoder (6 layers, 8 heads, 256-D)
- Or Bi-LSTM alternative for efficiency
- Processes phoneme sequences + style tags

### Generator
- Based on VITS architecture
- Cross-attention layers to condition on text
- Supports HiFiGAN or BigVGAN vocoder
- Stereo output options

### Training
- GAN-based with multi-period discriminator
- Losses: mel reconstruction, KL divergence, STFT, adversarial
- Mixed precision (FP16) for efficiency
- Adam optimizer with exponential LR decay

## Citation

Based on the research and development documented in:
"RVC V3: A Next-Generation Retrieval-Based Voice Conversion System"

Original RVC by RVC-Project: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

## License

Same license as the parent AudioLab project.

