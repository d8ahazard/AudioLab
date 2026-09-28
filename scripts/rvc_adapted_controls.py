"""Audition matched dataset adaptations and local Seed-VC fine-tunes."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, write_json, completed_job

def main():
    for case in json.loads((ROOT/'excerpts.json').read_text()):
        requests=[]
        if case['voice']=='shinedown':
            for variant in ('original','refreshed'):
                config=ROOT/'training'/variant/'run/config.json'
                if not config.exists(): continue
                cfg=json.loads(config.read_text()); epoch=cfg['train']['epochs']
                candidates=list((ROOT/'model_store/trained').glob(f'Shinedown666_{variant}_adapt_v{epoch}.pth'))
                if not candidates: continue
                if len(candidates)!=1: raise RuntimeError(f'Ambiguous export: {candidates}')
                requests.append((f'v2_{variant}_adapt',dict(backend='rvc_v2',checkpoint=str(candidates[0]),
                    index=str(ROOT/'models/Shinedown666_e75.index'),index_rate=0.,protect=.2,pitch_method='rmvpe+')))
        tune=ROOT/'seed_training'/case['voice']
        if (tune/'complete.json').exists():
            requests.append(('seed_vc_adapt',dict(backend='seed_vc',
                checkpoint=str(tune/'runs/adaptation/ft_model.pth'),config=str(tune/'config.yml'),
                reference=json.loads((ROOT/'references.json').read_text())[case['voice']]['audio']['path'],
                repository='E:/dev/AudioLab-experiments/seed-vc',steps=50)))
        for variant,request in requests:
            folder=ROOT/'runs'/case['id']/variant
            request.update(source=case['source']['path'],output_dir=str(folder),mode='preserve',seed=20260924)
            if completed_job(folder/'result.json',request): continue
            write_json(folder/'request.json',request)
            python=sys.executable if request['backend']=='rvc_v2' else 'D:/venvs/audiolab-svc/Scripts/python.exe'
            with (folder/'run.log').open('w') as log:
                subprocess.run([python,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('Converted',case['id'],variant,flush=True)

if __name__=='__main__':main()
