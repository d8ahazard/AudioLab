"""Focused existing-RVC versus singer-fine-tuned Seed-VC comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, fingerprint, write_json, completed_job
SVC='D:/venvs/audiolab-svc/Scripts/python.exe'


def prepare():
    import numpy as np
    import soundfile as sf
    dest=ROOT/'seed_training/chester/data'; dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for source in sorted(Path('D:/AI-outputs/AudioLab/voices/ChesterRedux/raw').glob('*.wav')):
        audio,sr=sf.read(source,dtype='float32',always_2d=True); audio=audio.mean(1)
        identity=fingerprint(source)
        for i,start in enumerate(range(0,len(audio),10*sr)):
            clip=audio[start:start+10*sr]
            if len(clip)<sr or np.sqrt(np.mean(clip**2))<.003:continue
            target=dest/f'{source.stem}_{i:03d}.wav'
            if not target.exists():sf.write(target,clip,sr,subtype='FLOAT')
            manifest.append(dict(source=str(source),source_identity=identity,start=start/sr,clip=fingerprint(target)))
    if not manifest:raise RuntimeError('No Chester training audio')
    write_json(dest.parent/'manifest.json',manifest)
    source=next(Path('D:/AI-outputs/AudioLab/voices/ChesterRedux/raw').glob('Faint*.wav'))
    audio,sr=sf.read(source,dtype='float32',always_2d=True);audio=audio.mean(1)
    starts=range(0,max(1,len(audio)-15*sr),sr)
    start=max(starts,key=lambda s:float(np.mean(audio[s:s+15*sr]**2)))
    target=dest.parent/'reference.wav'
    if not target.exists():sf.write(target,audio[start:start+15*sr],sr,subtype='FLOAT')
    write_json(dest.parent/'reference.json',dict(audio=fingerprint(target),source=fingerprint(source),start=start/sr))
    write_json(ROOT/'focused_seed_ab.json',dict(voices=['huxlxy','shinedown','chester'],
        comparisons='Existing RVC versus singer-fine-tuned Seed-VC only',passages_per_voice=2,
        v3='paused by user',chester_model=fingerprint('D:/models/AudioLab/trained/ChesterRedux.pth'),
        chester_test='The Emptiness Machine passages 0 and 2; not present in Chester training data',
        limitation='Shinedown Seed uses refreshed vocals; comparison measures the resulting system, not architecture alone. All Seed fine-tunes are 500-step candidates, not established optimal checkpoints.'))
    (ROOT/'v3/PAUSED').write_text('Paused by user: focus only on RVC versus trained Seed-VC for Chester, Huxlxy, Shinedown.')


def render():
    cases={x['id']:x for x in json.loads((ROOT/'excerpts.json').read_text())}
    refs=json.loads((ROOT/'references.json').read_text())
    for voice,prefix in [('huxlxy','hux'),('shinedown','shine'),('chester','chester')]:
        tune=ROOT/'seed_training'/voice
        trained=(tune/'complete.json').exists()
        reference=json.loads((tune/'reference.json').read_text())['audio']['path'] if voice=='chester' else refs[voice]['audio']['path']
        for part in (0,2):
            case=cases[f'{prefix}_test_{part}']; test=f'{voice}_{part}'
            for backend in ('rvc_v2','seed_vc'):
                if backend=='seed_vc' and not trained:continue
                folder=ROOT/'seed_ab/runs'/test/backend
                if backend=='rvc_v2':
                    if voice=='chester':
                        req=dict(backend=backend,checkpoint='D:/models/AudioLab/trained/ChesterRedux.pth',index='D:/models/AudioLab/trained/ChesterRedux.index',index_rate=.5,protect=.2,pitch_method='rmvpe+')
                    else:
                        variant='v2_r0.5_p0.5_rmvpe+' if voice=='shinedown' else 'v2_r0.5_p0.2_rmvpe+'
                        saved=ROOT/'runs'/case['id']/variant/'result.json'
                        result=json.loads(saved.read_text())
                        if not completed_job(saved,result['request']):raise RuntimeError(f'Stale baseline: {saved}')
                        write_json(folder/'result.json',result);continue
                else:
                    # Reuse verified Hux renders of this exact fine-tune and reference.
                    existing=ROOT/'runs'/case['id']/'seed_vc_adapt/result.json'
                    if voice=='huxlxy' and existing.exists():
                        result=json.loads(existing.read_text())
                        if completed_job(existing,result['request']) and result['request']['source']==case['source']['path'] and result['request']['reference']==reference and result['inputs']['checkpoint']['sha256']==fingerprint(tune/'runs/adaptation/ft_model.pth')['sha256']:
                            write_json(folder/'result.json',result);continue
                    req=dict(backend=backend,checkpoint=str(tune/'runs/adaptation/ft_model.pth'),config=str(tune/'config.yml'),reference=reference,repository='E:/dev/AudioLab-experiments/seed-vc',steps=50)
                req.update(source=case['source']['path'],output_dir=str(folder),seed=20260924,mode='preserve')
                if completed_job(folder/'result.json',req):continue
                write_json(folder/'request.json',req)
                with (folder/'run.log').open('w') as log:
                    subprocess.run([sys.executable if backend=='rvc_v2' else SVC,'-u',str(REPO/'scripts/rvc_backend_worker.py'),str(folder/'request.json')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
                print('Rendered',test,backend,flush=True)


def report():
    import numpy as np
    import soundfile as sf
    import librosa
    root=ROOT/'listening';cases={x['id']:x for x in json.loads((ROOT/'excerpts.json').read_text())}
    sections=[];reveal=[]
    for voice,prefix in [('huxlxy','hux'),('shinedown','shine'),('chester','chester')]:
        for passage,part in enumerate((0,2),1):
            if f'{prefix}_test_{part}' not in cases:continue
            case=cases[f'{prefix}_test_{part}'];test=f'{voice}_{part}'
            paths=[ROOT/'seed_ab/runs'/test/b/'result.json' for b in ('rvc_v2','seed_vc')]
            if not all(p.exists() for p in paths):continue
            results=[json.loads(p.read_text()) for p in paths]
            random.Random('seed-ab-v1-'+voice).shuffle(results)
            source,sr=sf.read(case['source']['path'],dtype='float32',always_2d=True);source=source.mean(1)
            sep=json.loads((ROOT/'separation'/f'{prefix}_test/complete.json').read_text())
            backing_path=next(x['path'] for x in sep['outputs'] if x['path'].endswith('__(Instrumental).wav'))
            with sf.SoundFile(backing_path) as stream:
                stream.seek(round(case['start']*stream.samplerate));back=stream.read(round(case['seconds']*stream.samplerate),dtype='float32',always_2d=True)
                if stream.samplerate!=sr:back=librosa.resample(back.T,orig_sr=stream.samplerate,target_sr=sr).T
            n=min(len(source),len(back));back=back[:n];source=source[:n];rms=np.sqrt(np.mean(source**2));pairs=[]
            for result in results:
                audio,rate=sf.read(result['output'],dtype='float32',always_2d=True);audio=audio.mean(1)
                if rate!=sr:audio=librosa.resample(audio,orig_sr=rate,target_sr=sr)
                audio*=rms/max(float(np.sqrt(np.mean(audio**2))),1e-8)
                audio=np.pad(audio,(0,max(0,n-len(audio))))[:n]
                pairs.append((audio,back+audio[:,None]))
            gain=min(1.,.98/max(float(np.abs(a).max()) for pair in pairs for a in pair));cards=[]
            for label,result,(solo,mix) in zip('AB',results,pairs):
                token=hashlib.sha256(('seed-ab-v1|'+test+'|'+label).encode()).hexdigest()[:12]
                for suffix,audio in [('',mix),('_solo',solo)]:sf.write(root/'audio'/f'{token}{suffix}.wav',audio*gain,sr,subtype='PCM_24')
                reveal.append(dict(test=test,label=label,result=result,token=token))
                cards.append(f'<article><h2>{label}</h2><audio controls preload="none" src="audio/{token}.wav"></audio><details><summary>Solo vocals</summary><audio controls preload="none" src="audio/{token}_solo.wav"></audio></details><button data-test="{test}" data-pick="{label}">Pick {label}</button></article>')
            name={'huxlxy':'Huxlxy','shinedown':'Shinedown','chester':'Chester Bennington'}[voice]
            song={'shinedown':'Cracks in my Calm','huxlxy':'Preacher Man','chester':'The Emptiness Machine'}[voice]
            sections.append(f'<section><h2>{name} · passage {passage}</h2><p>{song} · {case["start"]:.1f}s · 12 seconds</p><div class="grid">'+''.join(cards)+f'</div><button data-test="{test}" data-pick="Tie">No meaningful difference</button></section>')
    write_json(root/'seed-ab-reveal.json',reveal)
    page='''<!doctype html><meta charset="utf-8"><title>RVC versus trained Seed-VC</title><style>body{font:18px system-ui;background:#151a24;color:#eef4ff;max-width:950px;margin:35px auto;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:#222b3a;border-radius:12px;padding:22px}audio{width:100%;margin:16px 0}button,select{font:inherit;padding:12px;margin:10px 0;border-radius:8px;border:0;cursor:pointer}button{background:#92d5ff}.chosen{background:#a0e8b4}[hidden]{display:none!important}details{margin:12px 0}@media(max-width:600px){.grid{grid-template-columns:1fr}}</style><h1>Existing RVC vs trained Seed-VC</h1><p>Two passages per singer. Pick A or B, or call a tie. The letters are blinded.</p><label>Test <select id="test"></select></label>'''+''.join(sections)+'''<p id="summary"></p><button id="copy">Copy my picks</button><p id="copied"></p><script>const sections=[...document.querySelectorAll('section')],menu=document.getElementById('test');const storage='seed-ab-v1';let picks=JSON.parse(localStorage.getItem(storage)||'{}');sections.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.querySelector('h2').textContent;menu.append(o)});function show(){sections.forEach((s,i)=>s.hidden=i!==+menu.value);document.querySelectorAll('audio').forEach(a=>a.pause())}function update(){document.querySelectorAll('[data-pick]').forEach(b=>b.classList.toggle('chosen',picks[b.dataset.test]===b.dataset.pick));document.getElementById('summary').textContent=Object.entries(picks).map(([k,v])=>k+': '+v).join(' · ')}document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>{picks[b.dataset.test]=b.dataset.pick;localStorage.setItem(storage,JSON.stringify(picks));update();if(+menu.value<sections.length-1){menu.value=+menu.value+1;show()}});menu.onchange=show;document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);document.getElementById('copy').onclick=async()=>{const text=Object.entries(picks).map(([k,v])=>k+': '+v).join(', ');try{await navigator.clipboard.writeText(text);document.getElementById('copied').textContent='Copied — paste into chat.'}catch(e){document.getElementById('copied').textContent=text}};show();update();</script>'''
    (root/'seed-ab.html').write_text(page,encoding='utf-8')
    (root/'index.html').write_text(page,encoding='utf-8')
    print('Published',len(sections),'A/B tests',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render','report','finish']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='report':report()
    elif a.action=='render':render();report()
    else:
        while not (ROOT/'seed_training/shinedown/complete.json').exists():time.sleep(10)
        if not (ROOT/'seed_training/chester/complete.json').exists():
            with (ROOT/'train-seed-chester.log').open('w') as log:
                subprocess.run([SVC,'-u',str(REPO/'scripts/train_seed_quality.py'),'chester'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        while not (ROOT/'separation/chester_test/complete.json').exists():time.sleep(10)
        from scripts.benchmark_rvc_quality import excerpts
        excerpts(ROOT)
        render();report()
