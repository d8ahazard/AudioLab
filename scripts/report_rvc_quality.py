"""Build a local blinded audition with separate reveal key and diagnostic metrics."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, write_json

def main():
    import numpy as np
    import soundfile as sf
    import librosa
    from scipy.signal import correlate, correlation_lags
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--metrics',action='store_true'); a=p.parse_args()
    root=ROOT/'listening'; root.mkdir(exist_ok=True)
    clips=root/'audio'; clips.mkdir(exist_ok=True)
    references=json.loads((ROOT/'references.json').read_text())
    corpus=json.loads((ROOT/'corpus.json').read_text())['corpus']
    key={}; cards=[]; measurements=[]; group_index={}
    def groups(variant):
        memberships=[]
        baseline=variant=='v2_r0.5_p0.2_rmvpe+'
        if baseline or variant in {'seed_vc','seed_vc_adapt','yingmusic','vevo2','vevo2_target_style'}:memberships.append('engines')
        if variant.startswith('v2_r') or variant=='v2_epoch315':memberships.append('v2')
        if baseline or variant=='v2_old_stem':memberships.append('stems')
        if variant in {'v2_original_adapt','v2_refreshed_adapt','v2_r0.0_p0.2_rmvpe+'}:memberships.append('data')
        if baseline or variant.startswith('v3_'):memberships.append('lyrics')
        return memberships
    f0_model=None
    if a.metrics:
        import torch
        torch.set_num_threads(4)
        from modules.rvc.infer.lib.rmvpe import RMVPE
        f0_model=RMVPE('D:/models/AudioLab/rvc/rmvpe.pt',is_half=False,device='cuda')
    def f0(path):
        identity=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        dest=ROOT/'metrics_cache'/f'{identity}.npy'; dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            audio,_=librosa.load(path,sr=16000)
            np.save(dest,f0_model.infer_from_audio(audio,thred=.03))
        return np.load(dest)
    for case in json.loads((ROOT/'excerpts.json').read_text()):
        results=[]
        for path in sorted((ROOT/'runs'/case['id']).glob('*/result.json')):
            result=json.loads(path.read_text()); result['variant']=path.parent.name
            results.append(result)
        if not results: continue
        random.Random('rvc-blind-'+case['id']).shuffle(results)
        source,sr=sf.read(case['source']['path'],dtype='float32',always_2d=True)
        source=source.mean(1); rms=np.sqrt(np.mean(source**2))
        item=next(x for x in corpus if case['id'].startswith(x['id']))
        done=json.loads((ROOT/'separation'/item['id']/'complete.json').read_text())
        backing=next(x['path'] for x in done['outputs'] if x['path'].endswith('__(Instrumental).wav'))
        with sf.SoundFile(backing) as f:
            f.seek(int(case['start']*f.samplerate)); mix=f.read(int(case['seconds']*f.samplerate),dtype='float32',always_2d=True)
            if f.samplerate!=sr: mix=librosa.resample(mix.T,orig_sr=f.samplerate,target_sr=sr).T
        mix=mix[:len(source)]
        rows=[]; rendered=[]
        for i,result in enumerate(results):
            token=hashlib.sha256((case['id']+'|'+result['variant']).encode()).hexdigest()[:12]
            audio,rate=sf.read(result['output'],dtype='float32',always_2d=True)
            audio=audio.mean(1)
            if rate!=sr: audio=librosa.resample(audio,orig_sr=rate,target_sr=sr)
            audio=audio*rms/max(float(np.sqrt(np.mean(audio**2))),1e-8)
            # Pad/truncate for audition only: preserve timing and retain original renders unchanged.
            fixed=np.pad(audio,(0,max(0,len(source)-len(audio))))[:len(source)]
            combined=mix+fixed[:len(mix),None]
            rendered.append((token,fixed,combined))
            key[token]=dict(case=case['id'],label=chr(65+i),variant=result['variant'],result=result)
            group_index[token]=groups(result['variant'])
            metric=dict(case=case['id'],variant=result['variant'],duration_error_ms=1000*(result['duration']-len(source)/sr),
                        seconds=result['seconds'],peak_vram_bytes=result['peak_vram_bytes'])
            if a.metrics:
                src=f0(case['source']['path']); dst=f0(result['output']); n=min(len(src),len(dst)); src=src[:n];dst=dst[:n]
                joint=(src>0)&(dst>0); cents=np.abs(1200*np.log2(dst[joint]/src[joint]))
                metric.update(voiced_disagreement=float(np.mean((src>0)!=(dst>0))),
                    median_pitch_error_cents=float(np.median(cents)) if len(cents) else None,
                    pitch_error_over_100_cents=float(np.mean(cents>100)) if len(cents) else None)
                env=lambda x: librosa.feature.rms(y=x,frame_length=1024,hop_length=441)[0]
                e1=env(source); e2=env(fixed); cc=correlate(e2-e2.mean(),e1-e1.mean());lags=correlation_lags(len(e2),len(e1)); mask=np.abs(lags)<=50
                metric['envelope_lag_ms']=float(lags[mask][np.argmax(cc[mask])]*441/sr*1000)
            measurements.append(metric)
            rows.append(f'<article><h3>Sample {chr(65+i)}</h3><label>Solo <audio controls preload="none" src="audio/{token}.wav"></audio></label><label>In song <audio controls preload="none" src="audio/{token}_mix.wav"></audio></label>'+''.join(f'<label>{name} <select data-sample="{token}" data-axis="{axis}"><option value="">Unrated</option>'+''.join(f'<option>{v}</option>' for v in range(1,6))+'</select></label>' for axis,name in [('naturalness','Naturalness'),('identity','Target identity'),('delivery','Target delivery'),('intelligibility','Intelligibility'),('alignment','Song alignment')])+f'<textarea data-sample="{token}" data-axis="notes" placeholder="Notes"></textarea></article>')
        # One common playback scale preserves the matched accompaniment across all variants.
        gain=min(1.,.98/max(float(np.abs(x).max()) for _,solo,mix in rendered for x in (solo,mix)))
        for token,solo,mix in rendered:
            sf.write(clips/f'{token}.wav',solo*gain,sr,subtype='PCM_24'); sf.write(clips/f'{token}_mix.wav',mix*gain,sr,subtype='PCM_24')
        sourcefile=f'{case["id"]}_source.wav';sf.write(clips/sourcefile,source*gain,sr,subtype='PCM_24')
        ref=references.get(case['voice'])
        refhtml=''
        if ref:
            import shutil
            refname=case['voice']+'_reference.wav'; shutil.copy2(ref['audio']['path'],clips/refname)
            refhtml=f'<label>Target singer reference <audio controls preload="none" src="audio/{refname}"></audio></label>'
        cards.append(f'<section><h2>{html.escape(case["id"])} — {case["start"]:.2f}s</h2><label>Refreshed source <audio controls preload="none" src="audio/{sourcefile}"></audio></label>{refhtml}<p>Passage coverage: consonants, sustained notes, register changes, expression need listening verification.</p><div class="grid">'+''.join(rows)+'</div></section>')
    for item in corpus:
        if item['split']!='test':continue
        result_paths=sorted((ROOT/'full_songs'/item['id']).glob('*/result.json'))
        if not result_paths:continue
        done=json.loads((ROOT/'separation'/item['id']/'complete.json').read_text())
        lead=next(x['path'] for x in done['outputs'] if x['path'].endswith('__(Vocals).wav'))
        instrumental=next(x['path'] for x in done['outputs'] if x['path'].endswith('__(Instrumental).wav'))
        source,sr=sf.read(lead,dtype='float32',always_2d=True);source=source.mean(1)
        backing,back_sr=sf.read(instrumental,dtype='float32',always_2d=True)
        if back_sr!=sr:backing=librosa.resample(backing.T,orig_sr=back_sr,target_sr=sr).T
        n=min(len(source),len(backing));source=source[:n];backing=backing[:n]
        rms=float(np.sqrt(np.mean(source**2)));rendered=[];rows=[]
        random.Random('full-'+item['id']).shuffle(result_paths)
        for i,path in enumerate(result_paths):
            result=json.loads(path.read_text());token=hashlib.sha256(('full|'+item['id']+'|'+path.parent.name).encode()).hexdigest()[:12]
            audio,rate=sf.read(result['output'],dtype='float32',always_2d=True);audio=audio.mean(1)
            if rate!=sr:audio=librosa.resample(audio,orig_sr=rate,target_sr=sr)
            audio*=rms/max(float(np.sqrt(np.mean(audio**2))),1e-8)
            audio=np.pad(audio,(0,max(0,n-len(audio))))[:n]
            rendered.append((token,audio,backing+audio[:,None]))
            key[token]=dict(case=item['id']+'_full',label=chr(65+i),variant=path.parent.name,result=result)
            group_index[token]=['engines']
            rows.append(f'<article><h3>Complete song {chr(65+i)}</h3><label>Solo <audio controls preload="none" src="audio/{token}.wav"></audio></label><label>In song <audio controls preload="none" src="audio/{token}_mix.wav"></audio></label>'+''.join(f'<label>{axis} <select data-sample="{token}" data-axis="{axis}"><option value="">Unrated</option>'+''.join(f'<option>{v}</option>' for v in range(1,6))+'</select></label>' for axis in ('naturalness','identity','delivery','intelligibility','alignment'))+f'<textarea data-sample="{token}" data-axis="notes" placeholder="Full-song notes"></textarea></article>')
        gain=min(1.,.98/max(float(np.abs(x).max()) for _,solo,mix in rendered for x in (solo,mix)))
        for token,solo,mix in rendered:
            sf.write(clips/f'{token}.wav',solo*gain,sr,subtype='PCM_24');sf.write(clips/f'{token}_mix.wav',mix*gain,sr,subtype='PCM_24')
        cards.append(f'<section><h2>{html.escape(item["id"])} · complete-song confirmation</h2><p>Candidate renders; no winner has been selected automatically.</p><div class="grid">'+''.join(rows)+'</div></section>')
    write_json(root/'reveal-key.json',key); write_json(root/'metrics.json',measurements)
    page='''<!doctype html><meta charset="utf-8"><title>RVC blind listening</title><style>body{font:16px system-ui;background:#151a24;color:#eff2f8;max-width:1450px;margin:40px auto;padding:20px}h1,h2{color:#92d5ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{background:#222b3a;padding:20px;border-radius:12px}label{display:block;margin:12px 0}audio{display:block;width:100%;margin-top:8px}select,textarea,button{font:inherit;padding:8px}textarea{width:90%}button{position:sticky;top:10px;background:#92d5ff;border:0;border-radius:8px}section{margin:40px 0}</style><h1>Two voices · blinded conversion listening</h1><p>Score 1 (poor) to 5 (excellent). Leave unknown dimensions unrated. Solo vocals are RMS matched; every candidate uses the same instrumental and playback scale. Original renders are preserved. Delivery means resemblance to the target singer's performance habits, while melody and placement should remain intact.</p><p>These passages were screened by energy, not labeled by an expert. Two songs cannot establish broad superiority. Runtime includes loading and shared GPU activity. Do not use diagnostic metrics as perceptual rankings. Keep reveal-key.json closed until scoring.</p><button onclick="download()">Export listening scores</button>'''+''.join(cards)+'''<script>const controls=[...document.querySelectorAll('[data-sample]')];for(const c of controls){const k='rvc:'+c.dataset.sample+':'+c.dataset.axis;c.value=localStorage.getItem(k)||'';c.onchange=()=>localStorage.setItem(k,c.value)}function download(){const scores=controls.filter(c=>c.value).map(c=>({sample:c.dataset.sample,axis:c.dataset.axis,value:c.value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({date:new Date().toISOString(),scores},null,2)],{type:'application/json'}));a.download='rvc-listening-scores.json';a.click()}</script>'''
    navigation='<label>Experiment <select id="experiment"><option value="engines">Conversion engines</option><option value="v2">V2 settings and checkpoint</option><option value="stems">Old versus refreshed source</option><option value="data">Training data adaptation</option><option value="lyrics">V3 and lyric controls</option></select></label><label>Passage <select id="passage"></select></label><p id="availability"></p>'
    page=page.replace('<button onclick="download()">',navigation+'<button onclick="download()">')
    page+='<script>const groupIndex='+json.dumps(group_index)+''';const sections=[...document.querySelectorAll('section')];const passage=document.getElementById('passage');const experiment=document.getElementById('experiment');sections.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.querySelector('h2').textContent;passage.append(o)});function filter(){let visible=0;sections.forEach((s,i)=>{s.hidden=String(i)!==passage.value;s.querySelectorAll('article').forEach(a=>{const token=a.querySelector('[data-sample]').dataset.sample;a.hidden=!groupIndex[token].includes(experiment.value);if(!a.hidden&&!s.hidden)visible++})});document.getElementById('availability').textContent=visible?visible+' completed samples in this comparison.':'No completed samples in this comparison yet.';document.querySelectorAll('audio').forEach(a=>a.pause())}passage.onchange=filter;experiment.onchange=filter;filter();</script>'''
    # The four-test listening round is frozen; background sweeps must not replace it.
    destination=root/('diagnostics.html' if (root/'quick.html').exists() else 'index.html')
    destination.write_text(page,encoding='utf-8')
    print(destination)

if __name__=='__main__': main()
