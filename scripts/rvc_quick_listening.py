"""Four fixed listening tests, three candidates and one winner per test."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, REPO, write_json, completed_job


def main():
    import numpy as np
    import soundfile as sf
    import librosa
    root = ROOT/'listening'
    cases = {x['id']: x for x in json.loads((ROOT/'excerpts.json').read_text())}
    variants = ['v2_r0.75_p0.2_rmvpe+', 'v2_r0.5_p0.2_rmvpe+', 'v2_r0.5_p0.5_rmvpe+']
    tests = []
    # Freeze this round: future experiment renders must not add candidates or change labels.
    for number, case_id in enumerate(['hux_test_0', 'hux_test_2', 'shine_test_0', 'shine_test_2'], 1):
        case = cases[case_id]
        candidates = []
        for index, variant in enumerate(variants):
            folder = ROOT/'runs'/case_id/variant
            exemplar = ROOT/'runs'/case_id.replace('_2', '_0')/variant/'result.json'
            request = json.loads(exemplar.read_text())['request'].copy()
            request.update(source=case['source']['path'], output_dir=str(folder))
            if not completed_job(folder/'result.json', request):
                write_json(folder/'request.json', request)
                with (folder/'quick-run.log').open('w') as log:
                    subprocess.run([sys.executable, '-u', str(REPO/'scripts/rvc_backend_worker.py'), str(folder/'request.json')], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((folder/'result.json').read_text())
            candidates.append(dict(label=['B','C','F'][index] if number == 1 else chr(65+index), variant=variant, result=result))
        source, sr = sf.read(case['source']['path'], dtype='float32', always_2d=True)
        source = source.mean(1)
        item_id = case_id.rsplit('_', 1)[0]
        sep = json.loads((ROOT/'separation'/item_id/'complete.json').read_text())
        instrumental = next(x['path'] for x in sep['outputs'] if x['path'].endswith('__(Instrumental).wav'))
        with sf.SoundFile(instrumental) as stream:
            stream.seek(round(case['start']*stream.samplerate))
            backing = stream.read(round(case['seconds']*stream.samplerate), dtype='float32', always_2d=True)
            if stream.samplerate != sr:
                backing = librosa.resample(backing.T, orig_sr=stream.samplerate, target_sr=sr).T
        n = min(len(source), len(backing)); source=source[:n]; backing=backing[:n]
        rms = np.sqrt(np.mean(source**2)); rendered=[]
        for candidate in candidates:
            audio, rate = sf.read(candidate['result']['output'], dtype='float32', always_2d=True)
            audio=audio.mean(1)
            if rate != sr: audio=librosa.resample(audio,orig_sr=rate,target_sr=sr)
            audio *= rms/max(float(np.sqrt(np.mean(audio**2))),1e-8)
            audio=np.pad(audio,(0,max(0,n-len(audio))))[:n]
            rendered.append((audio, backing+audio[:,None]))
        gain=min(1.,.98/max(float(np.abs(a).max()) for pair in rendered for a in pair))
        for candidate, (solo,mix) in zip(candidates,rendered):
            token=hashlib.sha256(('quick-v1|'+case_id+'|'+candidate['variant']).encode()).hexdigest()[:12]
            candidate['token']=token
            for suffix,audio in [('',mix),('_solo',solo)]:
                sf.write(root/'audio'/f'{token}{suffix}.wav', audio*gain, sr, subtype='PCM_24')
        tests.append(dict(number=number,case=case_id,start=case['start'],candidates=candidates))
        print('Ready test',number,flush=True)
    write_json(root/'quick-reveal-key.json', tests)
    sections=[]
    for test in tests:
        number=test['number']; song='Huxlxy — Preacher Man' if number<3 else 'Shinedown — Cracks in my Calm'
        cards=[]
        for c in test['candidates']:
            label=c['label']; token=c['token']
            cards.append(f'<article><h2>{label}</h2><audio controls preload="none" src="audio/{token}.wav"></audio><details><summary>Solo vocals</summary><audio controls preload="none" src="audio/{token}_solo.wav"></audio></details><button data-test="{number}" data-choice="{label}">Pick {label}</button></article>')
        sections.append(f'<section data-number="{number}"><h2>Test {number} of 4 · {song}</h2><p>Passage {1 if number%2 else 2} · {test["start"]:.1f}s · 12 seconds</p><div class="grid">'+''.join(cards)+'</div></section>')
    page='''<!doctype html><meta charset="utf-8"><title>Four quick voice tests</title><style>body{font:18px system-ui;background:#151a24;color:#eff2f8;max-width:1100px;margin:40px auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{background:#222b3a;padding:22px;border-radius:12px}audio{width:100%;margin:15px 0}button{font:inherit;padding:12px 20px;cursor:pointer;border:0;border-radius:8px;background:#92d5ff}button.chosen{background:#a0e8b4}details{margin:15px 0}nav{display:flex;gap:12px}h1,h2{color:#92d5ff}[hidden]{display:none!important}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><h1>Four tests. Pick one per test.</h1><p>Two passages per song, three candidates each. Choose the voice you prefer overall.</p><p>Test 1 keeps your original B, C and F labels. Your earlier feedback is saved.</p><nav>'''+''.join(f'<button onclick="show({n})">Test {n}</button>' for n in range(1,5))+'''</nav>'''+''.join(sections)+'''<p id="summary"></p><button onclick="copyResults()">Copy my picks</button><p id="copied"></p><script>const key='rvc-quick-round-1';let picks=JSON.parse(localStorage.getItem(key)||'{}');function update(){document.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('chosen',picks[b.dataset.test]===b.dataset.choice));document.getElementById('summary').textContent=[1,2,3,4].map(n=>'Test '+n+': '+(picks[n]||'—')).join(' · ')}function show(n){document.querySelectorAll('audio').forEach(a=>a.pause());document.querySelectorAll('section').forEach(s=>s.hidden=+s.dataset.number!==n)}document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{picks[b.dataset.test]=b.dataset.choice;localStorage.setItem(key,JSON.stringify(picks));update();if(+b.dataset.test<4)show(+b.dataset.test+1)});document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);async function copyResults(){const text=[1,2,3,4].map(n=>'Test '+n+': '+(picks[n]||'unpicked')).join(', ');try{await navigator.clipboard.writeText(text);document.getElementById('copied').textContent='Copied — paste your picks into chat.'}catch(e){document.getElementById('copied').textContent=text}}update();show(1);</script>'''
    (root/'quick.html').write_text(page,encoding='utf-8')
    (root/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
