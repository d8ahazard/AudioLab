import soundfile as sf

from modules.diatts.dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# You should put the transcript of the voice you want to clone
# We will use the audio created by running simple.py as an example.
# Note that you will be REQUIRED TO RUN simple.py for the script to work as-is.
clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
clone_from_audio = "simple.mp3"

# For your custom needs, replace above with below and add your audio file to this directory:
# clone_from_text = "[S1] ... [S2] ... [S1] ... corresponding to your_audio_name.mp3"
# clone_from_audio = "your_audio_name.mp3"

# Text to generate
text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."

# It will only return the audio from the text_to_generate
output = model.generate(clone_from_text + text_to_generate, audio_prompt_path=clone_from_audio)

sf.write("voice_clone.mp3", output, 44100)
