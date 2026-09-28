"""Full-song confirmation renders for the fixed V2 baseline and singing backends."""
import json
from pathlib import Path
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, MODELS, write_json, completed_job

def main():
    for item in json.loads((ROOT/'corpus.json').read_text())['corpus']:
        if item['split']!='test': continue
        done=json.loads((ROOT/'separation'/item['id']/'complete.json').read_text())
        source=next(p['path'] for p in done['outputs'] if p['path'].endswith('__(Vocals).wav'))
        for variant in ('v2_r0.5_p0.2_rmvpe+','seed_vc','yingmusic','seed_vc_adapt'):
            exemplar=ROOT/'runs'/(item['id']+'_0')/variant/'result.json'
            if not exemplar.exists(): continue
            request=json.loads(exemplar.read_text())['request'].copy()
            folder=ROOT/'full_songs'/item['id']/variant
            request.update(source=source,output_dir=str(folder))
            if completed_job(folder/'result.json',request): continue
            write_json(folder/'request.json',request)
            python=sys.executable if request['backend']=='rvc_v2' else 'D:/venvs/audiolab-svc/Scripts/python.exe'
            with (folder/'run.log').open('w') as log:
                subprocess.run([python,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('Converted complete song',item['id'],variant,flush=True)

if __name__=='__main__':main()
