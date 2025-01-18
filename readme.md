

## Pre-requisites
1. Python. I use 3.10. 3.11 is probably fine too.
2. CUDA stuff. CU121.
3. Make a venv. This is documented to death, this is a grownup project, I believe in you.
4. Activate said venv before installation. Or don't. You'll probably have a bad time if you don't.

## Installing

1. If you have troubles on windoze - be sure PIP is < 24.1, as they added some new thing that breaks installation of most omegaconf installs.
````python.exe -m pip install pip==24.0````
2. CUDA drivers (Windoze) - You might need to download old CUDA drivers from Nvidiacudnn-11.2-windows-x64-v8.1.1.33.zip
3. Fairseq. WTF dude. fairseq>=0.12.2 --no-deps manually? Maybe.
4. The setup.bat and setup.sh scripts should work, but have only been tested against cu121.