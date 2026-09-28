"""Supported Seed-VC fine-tuning, isolated data/output and retained checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('voice',choices=['huxlxy','shinedown','chester'])
    p.add_argument('--steps',type=int,default=500)
    a=p.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    root=Path('D:/AI-outputs/AudioLab/rvc_quality_validation')
    repo=Path('E:/dev/AudioLab-experiments/seed-vc')
    os.chdir(repo); sys.path.insert(0,str(repo))
    import torch
    import numpy as np
    import yaml
    from huggingface_hub import hf_hub_download
    random.seed(20260924); np.random.seed(20260924); torch.manual_seed(20260924)
    torch.set_num_threads(4)
    checkpoint=hf_hub_download('Plachta/Seed-VC','DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth')
    source=hf_hub_download('Plachta/Seed-VC','config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
    config=yaml.safe_load(Path(source).read_text())
    config.update(log_dir=str(root/'seed_training'/a.voice/'runs'),keep_all_checkpoints=True)
    target=root/'seed_training'/a.voice/'config.yml'
    target.write_text(yaml.safe_dump(config))
    from train import Trainer
    trainer=Trainer(str(target),checkpoint,str(target.parent/'data'),'adaptation',
                    batch_size=2,num_workers=0,steps=a.steps,save_interval=100,device='cuda:0')
    trainer.train()
    (target.parent/'complete.json').write_text(json.dumps(dict(steps=a.steps,seed=20260924,
        initialization=checkpoint,config=str(target),status='candidate; listening required'),indent=2))

if __name__=='__main__': main()
