# AudioLab

![AudioLab Logo](./res/audiolab_lg.png)

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-cu121-brightgreen)](https://developer.nvidia.com/cuda-downloads)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)

> **Huge thanks to RunDiffusion for supporting this project!** 🎉

AudioLab is an open-source powerhouse for voice-cloning and audio separation, built with modularity and extensibility in mind. Whether you're an audio engineer, researcher, or just a curious tinkerer, AudioLab has you covered.

---

## 🌟 Features

### 🎵 Audio Processing Capabilities
- **🎼 Music Generation:** Create music from scratch or remix existing tracks using YuE.
- **🎵 Song Generation:** Create full-length songs with vocals and instrumentals using DiffRhythm.
- **🗣️ Zonos Text-to-Speech:** High-quality TTS with deep learning.
- **🎭 Orpheus TTS:** Real-time natural-sounding speech powered by large language models.
- **📢 Text-to-Speech:** Clone voices and generate natural-sounding speech with Coqui TTS.
- **🔊 Text-to-Audio:** Generate sound effects and ambient audio from text descriptions using Stable Audio.
- **🎛️ Audio Separation:** Isolate vocals, drums, bass, and other components from a track.
- **🎤 Vocal Isolation:** Distinguish lead vocals from background.
- **🔇 Noise Removal:** Get rid of echo, crowd noise, and unwanted sounds.
- **🧬 Voice Cloning:** Train high-quality voice models with just 30-60 minutes of data.
- **🚀 Audio Super Resolution:** Enhance and clean up audio.
- **🎚️ Remastering:** Apply spectral characteristics from a reference track.
- **🎵 Timbre Transfer:** Transform instrument sounds while preserving musical content using WaveTransfer.
- **🔄 Audio Conversion:** Convert between popular formats effortlessly.
- **📜 Export to DAW:** Easily create Ableton Live and Reaper projects from separated stems.

### 🤖 Automation Features
- **Auto-preprocessing** for voice model training.
- **Merge separated sources** back into a single file with ease.

---

## 🛠️ Pre-requisites

Before you dive in, make sure you have:

1. **Python 3.10** – *Because match statements exist, and fairseq is allergic to 3.11.*
2. **CUDA 12.4** – *Other versions? Maybe fine. Maybe not. Do you like surprises?*
3. **Virtual Environment** – *Strongly recommended to avoid dependency chaos.*
4. **Windows Users** – *You're in for an adventure! Zonos/Triton can be a pain. Make sure to install MSVC and add these paths to your environment variables:*
   ```plaintext
   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64
   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx86\x86
   ```

> **Note:** This project assumes basic Python knowledge. If you've never set up a virtual environment before... now's the time to learn! 🚀

---

## 🚑 Windows Troubleshooting

If dependencies refuse to install on Windows, try the following:

- Install **MSVC Build Tools**:
  - [VC Redist x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
  - [Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe)
- Ensure **CUDA is correctly installed**:
  - Check version: `nvcc --version`
  - [Download CUDA 12.4](https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe)
- DLL Errors? Try moving necessary DLLs from `/libs` to:
  ```plaintext
  .venv\lib\site-packages\pandas\_libs\window
  .venv\lib\site-packages\sklearn\.libs
  C:\Program Files\Python310\ (or wherever your Python is installed)
  ```

---

## 🚀 Installation

> **Heads up!** The `requirements.txt` is *not* complete on purpose. Use the setup scripts instead!

### 🛠 Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audiolab.git
   cd audiolab
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Run the setup script:
   ```bash
   ./setup.sh  # Windows: setup.bat
   ```

**Common Issues & Fixes:**
- Downgrade `pip` if installation fails:
  ```bash
  python -m pip install pip==24.0
  ```
- Install older CUDA drivers if needed: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Install `fairseq` manually if necessary:
  ```bash
  pip install fairseq>=0.12.2 --no-deps
  ```

---

## 🎛️ Running AudioLab

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate  # Windows: venv\Scripts\activate.bat
   ```
2. Run the application:
   ```bash
   python main.py
   ```
3. Optional flags:
   - `--listen` → Bind to `0.0.0.0` for remote access.
   - `--port PORT` → Specify a custom port.

---

## 📸 Screenshots

| ![Screenshot 1](./res/img/ss1_zonos.png) | ![Screenshot 2](./res/img/ss2_tts.png) |
|---------------------------------|---------------------------------|
| ![Screenshot 3](./res/img/ss3_yue.png) | ![Screenshot 4](./res/img/ss4_process.png) |
| ![Screenshot 5](./res/img/ss4_train.png) | |

---

## 💻 Key Features

### Sound Forge: Text-to-Audio Generation

Generate high-quality sound effects, ambient audio, and musical samples from text descriptions:

- **🔊 Text Prompting:** Create sounds by describing them in natural language.
- **⏱️ Variable Duration:** Generate audio up to 47 seconds long.
- **🎛️ Full Control:** Adjust parameters like inference steps and guidance scale.
- **🎭 Negative Prompts:** Specify what to avoid in your generated audio.
- **🎲 Multiple Variations:** Generate different versions of the same prompt.

Example prompts:
- "A peaceful forest ambience with birds chirping and leaves rustling"
- "An electronic beat with pulsing bass at 120 BPM"
- "A sci-fi spaceship engine humming"


### WaveTransfer: Instrument Timbre Transfer

Transform the sound characteristics of one instrument to another using diffusion models:

- **🎵 Preserve Musical Content:** Transform timbre while keeping the original musical composition intact.
- **🎸 Multi-instrument Support:** Transfer between any types of musical instruments.
- **🔄 Two-Step Process:** Easy-to-follow train-then-generate workflow for custom instruments.
- **⚙️ Flexible Configuration:** Adjust noise schedules and steps for different transfer qualities.
- **💾 Memory Optimization:** Use chunked processing for longer audio files.

Example applications:
- Transform a piano recording to sound like a guitar
- Create hybrid instruments with unique sound characteristics
- Convert acoustic instrument recordings to electronic sounds
- Experiment with novel timbres for music production

### Orpheus TTS: Real-time Speech Synthesis

Generate natural-sounding speech with LLM-powered text-to-speech capabilities:

- **⚡ Real-time Processing:** Instantaneous speech generation.
- **🗣️ Voice Cloning:** Create custom voice models from your recordings.
- **😀 Emotion Control:** Adjust speaking style for more expressive speech.
- **🌐 Multilingual Support:** Generate speech in multiple languages.
- **🎭 Style Variety:** Create different styles from a single voice model.

Example applications:
- Create audiobooks with natural narration
- Develop voice assistants with your own voice
- Generate voiceovers for videos and presentations
- Create accessible content for those with reading difficulties

### Transcribe: Advanced Speech-to-Text

Convert audio recordings to text with speaker identification and precise timing:

- **👥 Speaker Diarization:** Automatically identify and label different speakers.
- **⏱️ Word-Level Timestamps:** Create perfectly aligned text with audio timing.
- **🌍 Multilingual Support:** Transcribe content in multiple languages.
- **📊 Batch Processing:** Process multiple audio files in sequence.
- **📋 Multiple Output Formats:** Generate both JSON metadata and readable text.

Example applications:
- Create subtitles for videos with speaker labels
- Transcribe interviews and meetings with speaker attribution
- Generate searchable archives of audio content
- Create training data for voice and speech models

### Process Tab: Audio Processing Pipeline

The heart of AudioLab with modular audio processing through a chain of wrappers:

- **🔊 Separate:** Split audio into vocals, drums, bass, and other instruments.
- **🎤 Clone:** Apply voice conversion with trained models.
- **⚡ Remaster:** Enhance audio based on reference tracks.
- **🔬 Super Resolution:** Improve audio detail and clarity.
- **🔀 Merge:** Mix separate audio tracks with complete control.
- **🔄 Convert:** Change audio formats with customizable settings.

Example workflows:
- Extract vocals → Apply voice clone → Merge with original instruments
- Split song → Enhance each component → Remix with new levels
- Remaster old recordings using modern reference tracks

### RVC Training: Voice Model Creation

Train custom voice models for voice conversion and cloning:

- **🎯 One-Click Process:** Simplified training with automatic preprocessing.
- **⚙️ Advanced Options:** Fine-tune training for specific voice characteristics.
- **📊 Training Visualization:** Monitor progress in real-time.
- **🔄 Model Management:** Organize and share your trained voice models.

Example applications:
- Create virtual versions of your own voice
- Develop character voices for games or animations
- Restore or enhance historical recordings

---

## 🤝 Acknowledgements

AudioLab is powered by some fantastic open-source projects:
- 🎵 [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) – Core for audio separation.
- 🎚 [matchering](https://github.com/sergree/matchering) – Professional-grade remastering.
- 🔊 [versatile-audio-super-resolution](https://github.com/haoheliu/versatile_audio_super_resolution) – High-quality audio enhancement.
- 🎙 [Real-Time-Voice-Cloning](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) – Voice cloning.
- 🎶 [MVSEP-MDX23](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model) – Music separation.
- 📜 [WhisperX](https://github.com/m-bain/whisperX) – Audio transcription.
- 🗣 [Coqui TTS](https://github.com/coqui-ai/TTS) – State-of-the-art TTS.
- 🎼 [YuE](https://github.com/multimodal-art-projection/YuE) – Music generation.
- 🏆 [Zonos](https://github.com/Zyphra/Zonos) – High-quality TTS.
- 🔈 [Stable Audio](https://stability.ai/blog/stable-audio-open-1-0-free-text-to-audio-model) – Text-to-audio generation.
- 🎵 [DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) – Full-length song generation with latent diffusion.
- 🗣️ [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) – Real-time high-quality text-to-speech.
- 🎵 [WaveTransfer](https://github.com/tencent-ailab/bddm) – Instrument timbre transfer with diffusion.

---

## 🌟 Contribute

Want to help? Check out the [Contributing Guide](CONTRIBUTING.md)! 

---

## 📜 License

Licensed under MIT. See [LICENSE](LICENSE) for details.

---

Made with ❤️ by the AudioLab team. (AKA D8ahazard)
