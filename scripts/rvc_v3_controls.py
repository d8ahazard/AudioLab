"""Text-off V3 controls after transfer parity; no claim of trained text conditioning."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, MODELS, write_json, completed_job

for case in json.loads((ROOT/'excerpts.json').read_text()):
    voice=case['voice'];folder=ROOT/'runs'/case['id']/'v3_text_off'
    parity=json.loads((ROOT/'v3'/voice/'parity.json').read_text())
    index=MODELS/'huxlxy_v2_v500.index' if voice=='huxlxy' else ROOT/'models/Shinedown666_e75.index'
    request=dict(backend='v3',source=case['source']['path'],checkpoint=str(ROOT/'v3'/voice/'f0G48k.safetensors'),
                 index=str(index),output_dir=str(folder),mode='preserve',seed=20260924,index_rate=.5,
                 text_strength=0.,noise_scale=.66666)
    if completed_job(folder/'result.json',request): continue
    write_json(folder/'request.json',request)
    with (folder/'run.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                       cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Converted',case['id'],flush=True)
