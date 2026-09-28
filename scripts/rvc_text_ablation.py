"""Within-checkpoint V3 text on/off/mismatched lyric comparisons on held-out test audio."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT,REPO,MODELS,write_json,completed_job

def main():
    transcripts=json.loads((ROOT/'transcripts.json').read_text())
    for case in json.loads((ROOT/'excerpts.json').read_text()):
        checkpoint=ROOT/'v3_text'/case['voice']/'adapter_200.safetensors'
        if not checkpoint.exists():continue
        index=MODELS/'huxlxy_v2_v500.index' if case['voice']=='huxlxy' else ROOT/'models/Shinedown666_e75.index'
        for variant,strength,lyrics in [('off',0.,None),('on',.35,transcripts[case['id']]['text']),
                ('mismatched',.35,transcripts[case['voice']+'_reference']['text'])]:
            folder=ROOT/'runs'/case['id']/('v3_adapter_'+variant)
            request=dict(backend='v3',checkpoint=str(checkpoint),index=str(index),source=case['source']['path'],
                output_dir=str(folder),mode='preserve',seed=20260924,index_rate=.5,text_strength=strength,
                lyrics=lyrics,noise_scale=.66666,transcript_reviewed=False)
            if completed_job(folder/'result.json',request):continue
            write_json(folder/'request.json',request)
            with (folder/'run.log').open('w') as log:
                subprocess.run([sys.executable,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('Converted',case['id'],variant,flush=True)

if __name__=='__main__':main()
