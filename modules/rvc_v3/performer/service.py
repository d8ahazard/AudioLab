"""Isolated performer jobs; preserve mode delegates to the existing V2 worker."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from .contracts import PerformerRequest,write_json


def convert(source_audio,performer_profile,output_dir,**kwargs):
    kwargs.setdefault('guide_conditioning','audio_only')
    folder=Path(output_dir)/('performer_'+uuid.uuid4().hex[:12])
    request=PerformerRequest(source_audio,performer_profile,str(folder),**kwargs).validate()
    write_json(folder/'request.json',asdict(request))
    worker=Path(__file__).resolve().parents[3]/'scripts/performer_worker.py'
    with (folder/'run.log').open('w') as log:
        done=subprocess.run([sys.executable,'-u',str(worker.resolve()),str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT)
    if done.returncode:raise RuntimeError(f'Performer generation failed; see {folder / "run.log"}')
    return json.loads((folder/'result.json').read_text())['output']
