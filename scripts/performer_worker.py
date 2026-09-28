"""Serialized performer generation. Every failure is explicit; no silent substitution."""
import json
from pathlib import Path
import sys
import time
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import PerformerRequest,identity,write_json,words
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def run(req):
    import numpy as np
    import soundfile as sf
    import torch
    req.validate();torch.set_num_threads(4);torch.manual_seed(req.seed);np.random.seed(req.seed)
    folder=Path(req.output_dir);folder.mkdir(parents=True,exist_ok=True)
    profile=json.loads(Path(req.performer_profile).read_text())
    for key in ('checkpoint','index','reference'):
        if identity(profile[key]['path'])!=profile[key]:raise ValueError(f'Profile {key} changed; rebuild provenance')
    if req.delivery_mode=='preserve' or req.delivery_strength==0:
        from scripts.rvc_backend_worker import run as existing_v2
        baseline=dict(backend='rvc_v2',source=req.source_audio,checkpoint=profile['checkpoint']['path'],
            index=profile['index']['path'],seed=req.seed,output_dir=str(folder),mode='preserve',**profile['baseline_settings'])
        result=existing_v2(baseline);result.update(performer_stage='exact_v2_route',schema=1)
        return result
    if req.stage=='learned':
        raise RuntimeError('M2 is gated: learned planner/bridge has not passed M1 or been trained. No random-weight inference.')
    from modules.rvc_v3.performer.analysis import transcribe,phrase_windows,align_corrected
    source,sr=sf.read(req.source_audio,dtype='float32',always_2d=True);source=source.mean(1)
    analysis=transcribe(req.source_audio)
    if req.source_transcript and words(req.source_transcript)!=words(analysis['text']):
        write_json(folder/'source-analysis-original.json',analysis)
        analysis=align_corrected(req.source_audio,req.source_transcript,len(source)/sr)
    write_json(folder/'source-analysis.json',analysis)
    reference=transcribe(profile['reference']['path']);write_json(folder/'reference-analysis.json',reference)
    phrases=phrase_windows(analysis,len(source)/sr,req)
    if not reference['text'].strip():raise ValueError('Reference has no usable transcript')
    guide_request=dict(seed=req.seed,reference=profile['reference']['path'],reference_text=reference['text'],
        phrases=phrases,output_dir=str(folder/'guides'),conditioning=req.guide_conditioning)
    write_json(folder/'guide-request.json',guide_request)
    with (folder/'guide.log').open('w') as log:
        subprocess.run(['D:/venvs/audiolab-performer/Scripts/python.exe','-u',str(Path(__file__).with_name('performer_guide_worker.py')),str(folder/'guide-request.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
    guide_report=json.loads((folder/'guides/guides.json').read_text())
    guides=guide_report['outputs']
    if len(guides)!=len(phrases) or any(g['id']!=p['id'] for g,p in zip(guides,phrases)):
        raise RuntimeError('Guide/phrase alignment contract failed')
    # Verify generated words before investing in acoustic rendering.
    for guide,phrase in zip(guides,phrases):
        text=transcribe(guide['audio']['path']);write_json(folder/'guides'/f'{phrase["id"]}.transcript.json',text)
        guide['word_check']=dict(expected=phrase['text'],observed=text['text'],exact=words(text['text'])==words(phrase['text']),reviewed=False)
    from modules.rvc_v3.performer.render import FrozenRVCGuideRenderer
    renderer=FrozenRVCGuideRenderer(profile);out=np.zeros(round(len(source)/sr*renderer.rate),dtype='float32')
    if req.phrase_edits:
        # Unedited regions are explicitly preserved via the same V2 model, not source identity.
        from scripts.rvc_backend_worker import run as existing_v2
        baseline=existing_v2(dict(backend='rvc_v2',source=req.source_audio,checkpoint=profile['checkpoint']['path'],index=profile['index']['path'],seed=req.seed,output_dir=str(folder/'unedited_v2'),mode='preserve',**profile['baseline_settings']))
        import librosa
        audio,rate=sf.read(baseline['output'],dtype='float32')
        if rate!=renderer.rate:audio=librosa.resample(audio,orig_sr=rate,target_sr=renderer.rate)
        out[:min(len(audio),len(out))]=audio[:len(out)]
    observations=[]
    for phrase,guide in zip(phrases,guides):
        wav,rate=sf.read(guide['audio']['path'],dtype='float32')
        start,end=round(phrase['start']*sr),round(phrase['end']*sr)
        audio,target_sr,fit=renderer.render(source[start:end],sr,wav,rate)
        a=round(start/sr*target_sr);b=a+len(audio)
        if b>len(out):raise RuntimeError('Phrase exceeds original timeline')
        if req.phrase_edits:
            from modules.rvc_v3.performer.assembly import place_phrase
            fit['placement']=place_phrase(out,a,audio,target_sr,edited=True)
        else:
            # Tiny edge fades only; do not time-stretch finished waveforms to force a fit.
            edge=min(round(.005*target_sr),len(audio)//2)
            if edge:audio[:edge]*=np.linspace(0,1,edge);audio[-edge:]*=np.linspace(1,0,edge)
            out[a:b]=audio
        # Map automatic guide words through the actual timing allocation, in seconds.
        guide_text=json.loads((folder/'guides'/f'{phrase["id"]}.transcript.json').read_text())
        mapping=np.concatenate([np.linspace(r['start'],r['end']-1,r['frames']) for r in fit['runs']])
        timeline=np.arange(len(mapping))/fit['frame_rate']+phrase['start']
        aligned=[dict(text=w['text'],start=float(np.interp(w['start']*fit['frame_rate'],mapping,timeline)),
                      end=float(np.interp(w['end']*fit['frame_rate'],mapping,timeline))) for w in guide_text['words']]
        observations.append(dict(**phrase,fit=fit,guide=guide,planned_word_alignment=aligned))
    peak=float(abs(out).max());gain=1. if req.phrase_edits else min(1.,.98/max(peak,1e-8));out*=gain
    target=folder/'performer.wav';sf.write(target,out,renderer.rate,subtype='FLOAT')
    rate=renderer.rate
    del renderer
    import gc
    gc.collect();torch.cuda.empty_cache()
    final_analysis=transcribe(target);write_json(folder/'output-analysis.json',final_analysis)
    if req.phrase_edits:
        expected=observed=None
        word_status='edited_regions_require_review'
    else:
        expected=' '.join(p['text'] for p in phrases);observed=final_analysis['text']
        word_status='automatic_words_match' if words(expected)==words(observed) and all(g['word_check']['exact'] for g in guides) else 'word_review_required'
    import psutil
    return dict(schema=1,performer_stage='M1_untrained_guide_control',output=str(target),output_identity=identity(target),
        guide_report=guide_report,worker=identity(__file__),request=vars(req),
        quality_status=word_status,output_word_check=dict(expected=expected,observed=observed,reviewed=False),
        peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),process_memory=psutil.Process().memory_info()._asdict(),
        source=identity(req.source_audio),profile=identity(req.performer_profile),sample_rate=rate,
        frames=len(out),timeline_seconds=len(out)/rate,peak_before_gain=peak,gain=gain,phrases=observations,
        reference_reviewed=profile['reference_reviewed'],requires_listening=True,
        limitations=['Automatic source/reference/guide transcripts need review','Voicing-run timing is not learned pronunciation or phone alignment','No trained performer adapter; this is the feasibility control','No retrieval blend in expressive control'])


if __name__=='__main__':
    from filelock import FileLock
    job=Path(sys.argv[1]);req=PerformerRequest(**json.loads(job.read_text()));start=time.perf_counter()
    try:
        with FileLock('D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'):
            result=run(req)
        result['wall_seconds']=time.perf_counter()-start
        write_json(Path(req.output_dir)/'result.json',result)
    except Exception as exc:
        write_json(Path(req.output_dir)/'failure.json',dict(error=type(exc).__name__,message=str(exc),wall_seconds=time.perf_counter()-start))
        raise
