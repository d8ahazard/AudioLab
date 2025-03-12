# YuE - UI

Welcome to YuE - UI, an interface for the YuE music generation model.\
The focus is on creativity, with an incremental song generation workflow using batch, select and continue.\
Low VRAM 6-8 GB is supported as well!

The project uses YuE-exllamav2 for music generation [sgsdxzy/YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2).\
The interface was started from [alisson-anjos/YuE-Interface](https://github.com/alisson-anjos/YuE-Interface).\
The official YuE repo can be found here [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
 
## Features
- Batch, select & continue
- Rewind
- Stage 1 & 2 preview
- Visualize generated song segments
- Load & save

## UI preview
![preview ui](/preview.png)

## Installation
Windows:
```bash
git clone https://github.com/joeljuvel/YuE-UI/
cd YuE-UI
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
python -m venv venv
./venv/scripts/activate
pip install -r requirements.txt
```
Place your YuE exl2 models in the `models/` folder.\
After installation you can run `start_windows.bat` to start the gradio server & then open http://127.0.0.1:7860 in your browser to access the UI.


## Credits
- YuE Official [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- YuE-exllamav2 [sgsdxzy/YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2)
- YuE Interface [alisson-anjos/YuE-Interface](https://github.com/alisson-anjos/YuE-Interface)
