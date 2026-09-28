"""Verify remote LFS hashes; preserve and replace interrupted Vevo2 downloads."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import hashlib
import shutil
import json
from huggingface_hub import HfApi, hf_hub_download

root=Path('D:/models/AudioLab-svc/Vevo2')
info=HfApi().model_info('RMSnow/Vevo2',files_metadata=True)
records=[]
for item in info.siblings:
    p=root/item.rfilename
    if not p.exists() or not item.lfs: continue
    digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
    if digest is None:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        digest=h.hexdigest()
    if digest!=item.lfs.sha256:
        backup=Path('D:/models/AudioLab-svc/Vevo2-interrupted')/item.rfilename
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists(): shutil.copy2(p,backup)
        print('Repairing',item.rfilename,flush=True)
        hf_hub_download('RMSnow/Vevo2',item.rfilename,local_dir=root,force_download=True)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=item.lfs.sha256: raise RuntimeError(f'Hash mismatch after repair: {p}')
    else: print('Verified',item.rfilename,flush=True)
    records.append(dict(path=str(p),sha256=item.lfs.sha256))
Path('D:/AI-outputs/AudioLab/rvc_quality_validation/vevo-models.json').write_text(
    json.dumps(dict(repository='RMSnow/Vevo2',revision=info.sha,files=records),indent=2))
