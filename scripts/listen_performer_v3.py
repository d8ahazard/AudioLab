"""Publish immutable A/B pairs only after automatic checks or recorded manual review."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')
LISTEN=Path('D:/AI-outputs/AudioLab/rvc_quality_validation/listening')


def publish(round_id,review_disputed_words=False,presentation=''):
    import numpy as np
    import soundfile as sf
    import librosa
    if not round_id.replace('-','').replace('_','').isalnum():raise ValueError('Invalid round identifier')
    if presentation and not presentation.replace('-','').isalnum():raise ValueError('Invalid presentation identifier')
    destination=LISTEN/(f'performer-{round_id}'+('-'+presentation if presentation else ''))
    if destination.exists():raise ValueError('Listening round already exists; never replace scored candidates')
    staging=LISTEN/f'.performer-{round_id}-{uuid.uuid4().hex[:8]}'
    cases=json.loads((ROOT/'benchmark.json').read_text());pairs=[];excluded=[]
    rng=random.Random(20260924)
    for case in cases:
        folder=ROOT/'rounds'/round_id/case['id']
        paths=[folder/mode/'result.json' for mode in ('preserve','performer')]
        if not all(p.exists() for p in paths):excluded.append(dict(case=case['id'],reason='missing successful render'));continue
        reports=[json.loads(p.read_text()) for p in paths]
        manual=folder/'review.json'
        reviewed=json.loads(manual.read_text()) if manual.exists() else {}
        output_hash=reports[1]['output_identity']['sha256']
        word_review_required=reports[1].get('quality_status')!='automatic_words_match'
        if word_review_required and not (reviewed.get('words_correct') and reviewed.get('output_sha256')==output_hash):
            metrics_path=folder/'word-metrics.json'
            metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
            if not review_disputed_words or not metrics or metrics['performer']['audio']['sha256']!=output_hash:
                excluded.append(dict(case=case['id'],reason='word mismatch needs review'));continue
            # This is only admission to human review, never a word-correctness pass.
            if metrics['performer']['wer']>max(.2,metrics['baseline']['wer']+.1):
                excluded.append(dict(case=case['id'],reason='substantial automatic word degradation'));continue
        clips=[]
        for report in reports:
            record=report['output_identity']
            if identity(record['path'])['sha256']!=record['sha256']:raise ValueError('Candidate hash mismatch')
            audio,sr=sf.read(record['path'],dtype='float32',always_2d=True);audio=audio.mean(1)
            if not np.isfinite(audio).all() or not np.any(audio):raise ValueError('Broken candidate')
            # Listening-only normalization; original V2 output remains byte-for-byte intact.
            if sr!=48000:audio=librosa.resample(audio,orig_sr=sr,target_sr=48000);sr=48000
            clips.append((audio,sr))
        if clips[0][1]!=clips[1][1] or abs(len(clips[0][0])-len(clips[1][0]))>clips[0][1]*.05:
            excluded.append(dict(case=case['id'],reason='sample rate or timeline mismatch'));continue
        rms=[float(np.sqrt(np.mean(a*a))) for a,_ in clips]
        target=min(.08,*[r*.95/max(float(abs(a).max()),1e-9) for r,(a,_) in zip(rms,clips)])
        order=[0,1];rng.shuffle(order);mapping={}
        for label,index in zip('AB',order):
            audio,sr=clips[index];gain=target/max(rms[index],1e-9)
            digest=hashlib.sha256((case['id']+label+round_id).encode()).hexdigest()[:20]
            file=staging/f'{digest}.wav';file.parent.mkdir(parents=True,exist_ok=True)
            sf.write(file,audio*gain,sr,subtype='PCM_24')
            record=identity(file);record['path']=str((destination/file.name).resolve())
            mapping[label]=dict(audio=record,backend=['rvc_v2','m1_untrained_guide'][index],gain=gain,original=reports[index]['output_identity'])
        pairs.append(dict(case=case['id'],mapping=mapping,word_review_required=word_review_required))
    if not pairs:
        write_json(ROOT/'rounds'/round_id/'listening-exclusions.json',excluded)
        raise RuntimeError('No candidates passed internal screening; see listening-exclusions.json')
    manifest=dict(round=round_id,stage='M1 feasibility; no learned performer adapters',pairs=pairs,excluded=excluded,
                  source_lyrics_reviewed=False,allows_disputed_words_for_listening=review_disputed_words,
                  presentation=presentation,playback_format='PCM_24; original float renders retained')
    write_json(staging/'mapping.json',manifest)
    cards=[]
    for pair in pairs:
        case=html.escape(pair['case']);items=[]
        for letter,entry in pair['mapping'].items():
            items.append(f'<div><b>{letter}</b><audio controls preload="none" src="{Path(entry["audio"]["path"]).name}"></audio><label><input type="radio" name="{case}" value="{letter}"> Pick {letter}</label></div>')
        cards.append(f'<article><h2>{case}</h2>'+''.join(items)+f'<input class="reason" data-case="{case}" placeholder="Optional reason"></article>')
    page='''<!doctype html><meta charset="utf-8"><title>Performer V3 feasibility</title>
<style>body{font:17px system-ui;background:#10131a;color:#edf2f9;max-width:900px;margin:40px auto;padding:20px}article{background:#1c2230;padding:22px;border-radius:12px;margin:20px 0}article div{display:flex;align-items:center;gap:18px;margin:16px 0}audio{flex:1}input.reason{width:90%;padding:10px}button{padding:12px}textarea{width:100%;height:110px}</style>
<h1>Performer V3 — first feasibility comparison</h1><p>Existing RVC versus the untrained pronunciation-guide control. This round does not claim learned artist delivery. Two passages per completed target, A/B only.</p>
<p>Pick the better overall vocal. Consider words, naturalness, identity and fit; a brief reason is optional. Automatic transcripts disagree on some words, including in the RVC baseline. This is a listening review, not a passed lyric-accuracy test; mention words you hear added, missing or wrong.</p>'''+''.join(cards)+'''<button id="copy">Copy choices</button><textarea id="choices" readonly></textarea>
<script>const key=location.pathname;const saved=JSON.parse(localStorage.getItem(key)||'{}');document.querySelectorAll('input').forEach(x=>{if(x.type==='radio')x.checked=saved[x.name]===x.value;else x.value=saved[x.dataset.case+'_reason']||'';x.onchange=save;});function save(){const data={};document.querySelectorAll('input:checked').forEach(x=>data[x.name]=x.value);document.querySelectorAll('.reason').forEach(x=>data[x.dataset.case+'_reason']=x.value);localStorage.setItem(key,JSON.stringify(data));document.getElementById('choices').value=Object.keys(data).filter(k=>!k.endsWith('_reason')).map(k=>k+': '+data[k]+(data[k+'_reason']?' ('+data[k+'_reason']+')':'')).join(' · ');}save();document.getElementById('copy').onclick=()=>navigator.clipboard.writeText(document.getElementById('choices').value);document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()}));</script>'''
    (staging/'index.html').write_text(page,encoding='utf-8')
    staging.rename(destination)
    print(f'http://127.0.0.1:8769/{destination.name}/')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--round',required=True)
    parser.add_argument('--review-disputed-words',action='store_true')
    parser.add_argument('--presentation',default='')
    args=parser.parse_args();publish(args.round,args.review_disputed_words,args.presentation)
