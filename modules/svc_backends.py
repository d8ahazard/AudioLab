"""Isolated offline singing backends; RVC continues using its existing runtime."""
import json
import os
from pathlib import Path
import subprocess
import uuid

REPOSITORIES={
    'seed_vc':'seed-vc', 'yingmusic':'YingMusic-SVC', 'vevo2':'Amphion',
}

def convert(source, reference, output_dir, backend='seed_vc', checkpoint=None,
            config=None, lyrics=None, reference_text=None, seed=20260924, mode='preserve'):
    if backend not in REPOSITORIES: raise ValueError(f'Unknown backend: {backend}')
    if mode not in ('preserve','target_style'): raise ValueError(f'Unknown conversion mode: {mode}')
    if mode=='target_style' and backend!='vevo2':
        raise ValueError('Target delivery mode currently requires Vevo2.')
    if mode=='target_style' and (not lyrics or not reference_text):
        raise ValueError('Vevo2 target delivery requires the source lyrics and reference transcript.')
    if lyrics and backend!='vevo2': raise ValueError('This backend does not accept lyric conditioning.')
    for path in (source,reference):
        if not path or not Path(path).is_file(): raise FileNotFoundError(f'Audio not found: {path}')
    interpreter=Path(os.environ.get('AUDIOLAB_SVC_PYTHON','D:/venvs/audiolab-svc/Scripts/python.exe'))
    repo=Path(os.environ.get('AUDIOLAB_SVC_REPOS','E:/dev/AudioLab-experiments'))/REPOSITORIES[backend]
    if not interpreter.is_file() or not repo.is_dir():
        raise RuntimeError('Singing backend is not installed. Configure AUDIOLAB_SVC_PYTHON and AUDIOLAB_SVC_REPOS.')
    if backend!='vevo2':
        from huggingface_hub import hf_hub_download
        if backend=='seed_vc':
            checkpoint=checkpoint or hf_hub_download('Plachta/Seed-VC','DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth')
            config=config or hf_hub_download('Plachta/Seed-VC','config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
        else:
            checkpoint=checkpoint or hf_hub_download('GiantAILab/YingMusic-SVC','YingMusic-SVC-full.pt')
            config=config or str(repo/'configs/YingMusic-SVC.yml')
        for path in (checkpoint,config):
            if not Path(path).is_file(): raise FileNotFoundError(path)
    elif checkpoint or config:
        raise ValueError('Vevo2 uses its complete model bundle; custom checkpoint/config overrides are unsupported.')
    folder=Path(output_dir)/f'{Path(source).stem}_{backend}_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True)
    request=dict(backend=backend,source=str(Path(source).resolve()),reference=str(Path(reference).resolve()),
                 output_dir=str(folder.resolve()),repository=str(repo.resolve()),checkpoint=checkpoint,
                 config=config,lyrics=lyrics,reference_text=reference_text,seed=int(seed),mode=mode,
                 steps=32 if backend=='vevo2' else 50)
    job=folder/'request.json'; job.write_text(json.dumps(request,indent=2),encoding='utf-8')
    worker=Path(__file__).resolve().parents[1]/'scripts/rvc_backend_worker.py'
    with (folder/'run.log').open('w') as log:
        result=subprocess.run([str(interpreter),'-u',str(worker),str(job)],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{backend} conversion failed. Details: {folder / "run.log"}')
    return json.loads((folder/'result.json').read_text())['output']
