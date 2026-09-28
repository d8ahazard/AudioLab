"""Same checkpoint/settings, old versus refreshed test stems, and Hux epoch 315."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, MODELS, write_json, fingerprint, completed_job

def main():
    import soundfile as sf
    import numpy as np
    corpus=json.loads((ROOT/'corpus.json').read_text())['corpus']
    for case in json.loads((ROOT/'excerpts.json').read_text()):
        item=next(x for x in corpus if case['id'].startswith(x['id']))
        old=item['old_lead']['path']
        audio,sr=sf.read(old,dtype='float32',always_2d=True)
        clip=ROOT/'excerpts'/f"{case['id']}_old.wav"
        samples=audio[int(case['start']*sr):int((case['start']+case['seconds'])*sr)].mean(1)
        same=False
        if clip.exists():
            prior,prior_sr=sf.read(clip,dtype='float32');same=prior_sr==sr and np.array_equal(prior,samples)
        if not same:sf.write(clip,samples,sr,subtype='FLOAT')
        hux=case['voice']=='huxlxy'
        model=MODELS/'huxlxy_v2_v500.pth' if hux else ROOT/'models/Shinedown666_e75.pth'
        index=MODELS/'huxlxy_v2_v500.index' if hux else ROOT/'models/Shinedown666_e75.index'
        variants=[('v2_old_stem',clip,model,index)]
        if hux:
            variants.append(('v2_epoch315',Path(case['source']['path']),MODELS/'huxlxy_v2.pth',MODELS/'huxlxy_v2.index'))
        for name,source,checkpoint,retrieval in variants:
            folder=ROOT/'runs'/case['id']/name
            request=dict(backend='rvc_v2',checkpoint=str(checkpoint),index=str(retrieval),
                         source=str(source),output_dir=str(folder),index_rate=.5,protect=.2,
                         pitch_method='rmvpe+',seed=20260924,mode='preserve')
            if completed_job(folder/'result.json',request): continue
            write_json(folder/'request.json',request)
            with (folder/'run.log').open('w') as log:
                subprocess.run([sys.executable,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                               cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            write_json(folder/'comparison.json',dict(control='v2_r0.5_p0.2_rmvpe+',
                original=fingerprint(old),start=case['start'],note='Identical source timeline; separator latency must be checked.'))

if __name__=='__main__': main()
