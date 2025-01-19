## What is this?
AudioLab is an open-source solution aimed to bridge the gap in voice-cloning and audio separation technologies in an 
easy-to-use, modular, and extensible way. It is designed to be a one-stop-shop for all your audio processing needs, within reason.

## What can it do?
Presently, AudioLab can:
1. Separate an input audio file into its constituent sources (vocals, drums, bass, other).
2. Separate background vocals from the lead vocals in an input audio file.
3. Remove reverb, echo, delay, crowd noise, and other background noise from an input audio file.
4. Generate a high-quality voice model from 30-60 minutes of audio data.
5. Automatically separate/clean input audio data for use in voice model training.
6. Apply "Audio Super Resolution" to input audio data, increasing the quality of the audio for output stems or before cloning a voice.
7. Clone a voice from a trained voice model to an input audio file.
8. Apply "Matchering" remastering to an output audio file, emulating the spectral characteristics of a reference audio file.
9. Automatically merge separated audio sources back into a single audio file.
10. Convert audio files between different formats.
11. Transcribe audio files to text.
12. Provide a spectral comparison between two audio files.

## Pre-requisites
1. Python. I use 3.10. 3.11 is probably fine too. 3.9 and lower is *not* supported, as I use "match" statements.
2. CUDA stuff. CU121. Others are probably fine, but I haven't tested them, and the requirements file is set to cu121.
3. Make a venv. This is documented to death, this is a grownup project, I believe in you.
4. Activate said venv before installation. Or don't. You'll probably have a bad time if you don't.

## Installing
1. If you have troubles on windoze - be sure PIP is < 24.1, as they added some new thing that breaks installation of most omegaconf installs.
````python.exe -m pip install pip==24.0````
2. CUDA drivers (Windoze) - You might need to download old CUDA drivers from Nvidiacudnn-11.2-windows-x64-v8.1.1.33.zip
3. Fairseq. WTF dude. fairseq>=0.12.2 --no-deps manually? Maybe.
4. The setup.bat and setup.sh scripts should work, but have only been tested against cu121.

## Running
1. Activate your venv.
2. Run python main.py. To enable listening on 0.0.0.0, use the --listen flag. To specify a port, use the --port PORT flag.

## Source Projects
AudioLab is built on the backs of giants. The following projects are Frankenstein-ed into AudioLab:
1. [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - Used for audio separation (But not all of it).
2. [matchering](https://github.com/sergree/matchering) - Used for audio remastering.
3. [versatile-audio-super-resolution](https://github.com/haoheliu/versatile_audio_super_resolution) - Used for audio super resolution.
4. [Real-Time-Voice-Cloning](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Used for voice cloning.
5. [ZFTurbo/MVSEP-MDX23-music-separation-model](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model) - Also used for audio separation.
6. [WhisperX](https://github.com/m-bain/whisperX) - For Transcription.