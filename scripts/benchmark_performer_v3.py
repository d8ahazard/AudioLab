"""Run one immutable M1 round: two passages per target, existing V2 versus guide control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')
OLD=Path('D:/AI-outputs/AudioLab/rvc_quality_validation')
REPO=Path(__file__).resolve().parents[1]


def freeze_cases():
    target=ROOT/'benchmark.json'
    if target.exists():
        cases=json.loads(target.read_text())
        for case in cases:
            if identity(case['source']['path'])!=case['source']:raise ValueError('Frozen benchmark source changed')
        return cases
    cases=[]
    for voice,prefix,root in [('tupac','tupac_test',ROOT),('huxlxy','hux_test',OLD),('shinedown','shine_test',OLD),('chester','chester_test',OLD)]:
        for passage in (0,2):
            source=root/'excerpts'/f'{prefix}_{passage}.wav'
            cases.append(dict(id=f'{voice}_{passage}',voice=voice,source=identity(source),
                              profile=identity(ROOT/'profiles'/f'{voice}.json'),role='development demonstration',
                              held_out_generalization=False))
    write_json(target,cases)
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',required=True)
    parser.add_argument('--voices',nargs='+',default=['tupac','huxlxy','shinedown','chester'])
    parser.add_argument('--passages',nargs='+',type=int,default=[0,2])
    parser.add_argument('--guide-conditioning',choices=['transcript','audio_only'],default='transcript')
    parser.add_argument('--modes',nargs='+',choices=['preserve','performer'],default=['preserve','performer'])
    args=parser.parse_args()
    if not args.round.replace('-','').replace('_','').isalnum():raise ValueError('Round name must be a simple identifier')
    cases=freeze_cases()
    for case in cases:
        if case['voice'] not in args.voices or int(case['id'].rsplit('_',1)[1]) not in args.passages:continue
        if identity(case['profile']['path'])!=case['profile']:raise ValueError('Profile changed; freeze a new benchmark version')
        for mode in args.modes:
            folder=ROOT/'rounds'/args.round/case['id']/mode
            request=dict(source_audio=case['source']['path'],performer_profile=case['profile']['path'],
                         output_dir=str(folder),delivery_mode=mode,seed=20260924,stage='m1_guide')
            if args.guide_conditioning!='transcript':request['guide_conditioning']=args.guide_conditioning
            request_path=folder/'request.json'
            if request_path.exists():
                if json.loads(request_path.read_text())!=request:raise ValueError('Immutable round request changed')
                if (folder/'result.json').exists():
                    result=json.loads((folder/'result.json').read_text());record=result['output_identity']
                    if identity(record['path'])['sha256']!=record['sha256']:raise ValueError('Scored candidate audio changed')
                    continue
                raise RuntimeError(f'Attempt already exists without success: {folder}. Use a new round ID for retries.')
            write_json(request_path,request)
            with (folder/'run.log').open('w') as log:
                proc=subprocess.run([sys.executable,'-u',str(REPO/'scripts/performer_worker.py'),str(request_path)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            result_path=folder/('result.json' if proc.returncode==0 else 'failure.json')
            event=dict(round=args.round,case=case['id'],mode=mode,status='rendered' if proc.returncode==0 else 'failed',report=str(result_path))
            with (ROOT/'results.jsonl').open('a',encoding='utf-8') as ledger:ledger.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
            if proc.returncode:raise RuntimeError(f'Feasibility failed: {result_path}. Diagnose before continuing.')


if __name__=='__main__':main()
