"""Controlled timing repair: reuse identical guides and baselines in a new immutable round."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json,words
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def run(parent_path,folder):
    import numpy as np
    import soundfile as sf
    import torch
    import gc
    from modules.rvc_v3.performer.render import FrozenRVCGuideRenderer
    from modules.rvc_v3.performer.analysis import transcribe
    started=time.perf_counter();torch.set_num_threads(4)
    previous=json.loads(parent_path.read_text());torch.manual_seed(previous['request']['seed']);torch.cuda.reset_peak_memory_stats()
    for record in [previous['source'],previous['profile']]+[p['guide']['audio'] for p in previous['phrases']]:
        if identity(record['path'])!=record:raise ValueError('Cached input changed')
    if previous['request'].get('phrase_edits'):raise ValueError('Timing control is for unchanged-lyric benchmarks')
    profile=json.loads(Path(previous['profile']['path']).read_text())
    for key in ('checkpoint','index','reference'):
        if identity(profile[key]['path'])!=profile[key]:raise ValueError('Profile asset changed')
    source,sr=sf.read(previous['source']['path'],dtype='float32',always_2d=True);source=source.mean(1)
    renderer=FrozenRVCGuideRenderer(profile);rate=renderer.rate
    output=np.zeros(round(len(source)/sr*rate),dtype='float32');observations=[]
    for original in previous['phrases']:
        phrase=copy.deepcopy(original);guide,gsr=sf.read(phrase['guide']['audio']['path'],dtype='float32')
        a,b=round(phrase['start']*sr),round(phrase['end']*sr)
        audio,_,fit=renderer.render(source[a:b],sr,guide,gsr)
        edge=min(round(.005*rate),len(audio)//2)
        if edge:audio[:edge]*=np.linspace(0,1,edge);audio[-edge:]*=np.linspace(1,0,edge)
        start=round(a/sr*rate)
        if start+len(audio)>len(output):raise ValueError('Timeline overflow')
        output[start:start+len(audio)]=audio
        phrase['fit']=fit
        guide_text=json.loads(Path(phrase['guide']['audio']['path']).with_suffix('.transcript.json').read_text())
        mapping=np.concatenate([np.linspace(r['start'],r['end']-1,r['frames']) for r in fit['runs']])
        timeline=np.arange(len(mapping))/fit['frame_rate']+phrase['start']
        phrase['planned_word_alignment']=[dict(text=w['text'],start=float(np.interp(w['start']*fit['frame_rate'],mapping,timeline)),
            end=float(np.interp(w['end']*fit['frame_rate'],mapping,timeline))) for w in guide_text['words']]
        observations.append(phrase)
    peak=float(abs(output).max());gain=min(1.,.98/max(peak,1e-8));output*=gain
    target=folder/'performer.wav';sf.write(target,output,rate,subtype='FLOAT')
    del renderer;gc.collect();torch.cuda.empty_cache()
    analysis=transcribe(target);write_json(folder/'output-analysis.json',analysis)
    expected=' '.join(p['text'] for p in observations)
    passed=words(expected)==words(analysis['text']) and all(p['guide']['word_check']['exact'] for p in observations)
    result=copy.deepcopy(previous)
    import psutil
    # Original generation cost belongs to the cached parent, not this render-only run.
    result.update(output=str(target),output_identity=identity(target),phrases=observations,
        request={**previous['request'],'output_dir':str(folder)},cached_parent=identity(parent_path),worker=identity(__file__),
        timing_algorithm='bounded_proportional_v2',quality_status='automatic_words_match' if passed else 'word_review_required',
        output_word_check=dict(expected=expected,observed=analysis['text'],reviewed=False),
        gain=gain,peak_before_gain=peak,wall_seconds=time.perf_counter()-started,
        peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),process_memory=psutil.Process().memory_info()._asdict())
    write_json(folder/'result.json',result)


def main():
    from filelock import FileLock
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True);parser.add_argument('--round',required=True)
    args=parser.parse_args()
    for value in (args.parent,args.round):
        if not value.replace('-','').replace('_','').isalnum():raise ValueError('Invalid round identifier')
    for case in json.loads((ROOT/'benchmark.json').read_text()):
        parent=ROOT/'rounds'/args.parent/case['id'];dest=ROOT/'rounds'/args.round/case['id'];folder=dest/'performer'
        if folder.exists():raise ValueError('Attempt already exists; use a new round ID')
        folder.mkdir(parents=True)
        write_json(dest/'preserve/result.json',json.loads((parent/'preserve/result.json').read_text()))
        try:
            with FileLock('D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'):run(parent/'performer/result.json',folder)
            event=dict(round=args.round,case=case['id'],mode='performer',status='rendered',report=str(folder/'result.json'))
        except Exception as exc:
            write_json(folder/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
            event=dict(round=args.round,case=case['id'],mode='performer',status='failed',report=str(folder/'failure.json'))
        with (ROOT/'results.jsonl').open('a') as ledger:ledger.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)


if __name__=='__main__':main()
