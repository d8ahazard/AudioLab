"""Resume finite evaluation stages; keep training and conversion jobs independently logged."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, write_json

def command(label, script, *args, python=sys.executable):
    state=ROOT/'stages'/f'{label}.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='complete': return
    write_json(state,dict(status='running',started=time.time(),command=[script,*args]))
    with (ROOT/f'{label}.log').open('a') as log:
        result=subprocess.run([python,'-u',str(REPO/'scripts'/script),*args],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    write_json(state,dict(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,finished=time.time()))
    if result.returncode: raise RuntimeError(f'{label} failed; see {ROOT / (label+".log")}')

def main():
    if (ROOT/'focused_seed_ab.json').exists():
        raise SystemExit('Broad evaluation paused for the user-requested RVC versus trained Seed-VC A/B study.')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lane',choices=['comparisons','refreshed']);a=p.parse_args()
    if a.lane=='comparisons':
        command('v2-resume','benchmark_rvc_quality.py','v2')
        command('external-seed-resume','benchmark_rvc_quality.py','seed_vc')
        command('external-ying-resume','benchmark_rvc_quality.py','yingmusic')
        command('external-vevo-resume','benchmark_rvc_quality.py','vevo2')
        command('external-vevo-style-resume','benchmark_rvc_quality.py','vevo2_style')
        command('v3-resume','rvc_v3_controls.py')
        command('stem-control-resume','rvc_stem_controls.py')
        command('full-song-controls','rvc_full_song_controls.py')
        command('report-resume','report_rvc_quality.py','--metrics')
    else:
        items=json.loads((ROOT/'corpus.json').read_text())['corpus']
        required=[ROOT/'separation'/x['id']/'complete.json' for x in items if x['voice']=='shinedown' and x['split']!='test']
        while not all(p.exists() for p in required): time.sleep(15)
        command('train-refreshed','train_rvc_quality.py','refreshed')
        command('seed-data-refreshed','train_rvc_quality.py','seed-data')
        command('train-seed-shine','train_seed_quality.py','shinedown',python='D:/venvs/audiolab-svc/Scripts/python.exe')
        command('adapted-controls','rvc_adapted_controls.py')
        command('full-song-adapted','rvc_full_song_controls.py')
        command('dereverb-controls','rvc_dereverb_controls.py')
        command('report-adapted','report_rvc_quality.py','--metrics')

if __name__=='__main__':main()
