"""Compare backing-vocal approaches on the listener-approved cleaned vocals."""
import json
import logging
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.separator.model_runtime import AudioLabSeparator, BG_MODELS, read_outputs
from modules.separator.stem_manifest import file_hash, model_fingerprint, write_json

ROOT = REPO / 'outputs/separation_v4_validation'
OUT = ROOT / 'background-comparison'


def read(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == 44100 and audio.shape[1] == 2 and np.isfinite(audio).all()
    return audio


def write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 44100, subtype='FLOAT')
    assert np.array_equal(read(path), audio.astype(np.float32))


def rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def main():
    separator = AudioLabSeparator(model_file_dir=str(REPO / 'models/audio_separator'),
        use_soundfile=True, use_autocast=True, log_level=logging.ERROR, quality='balanced', preserve_gain=True)
    sources = sorted((ROOT / 'hybrid-comparison/audio').glob('*/cleaned-vocals.wav'))
    assert len(sources) == 6
    for name, model in BG_MODELS.items():
        loaded = False
        for source in sources:
            folder = OUT / 'predictions' / source.parent.name / name
            cache = folder / 'complete.json'
            identity = {'source_sha256': file_hash(source), 'model': model, 'revision': 'bg-comparison-1',
                        'settings': {'quality': 'balanced', 'preserve_gain': True, 'autocast': True}}
            if cache.exists():
                record = json.loads(cache.read_text())
                if record['identity'] == identity and record['model_hashes'] == model_fingerprint(separator.model_file_dir, [model]) and all(file_hash(folder / f'{r}.wav') == h for r, h in record['outputs'].items()):
                    continue
            if not loaded:
                separator.load_model(model)
                loaded = True
            folder.mkdir(parents=True, exist_ok=True)
            audio = read(source)
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                separator.output_dir = temp
                separator.model_instance.output_dir = temp
                stems = read_outputs(separator.separate(str(source)), temp, 44100, len(audio), 'karaoke' if name == 'karaoke' else 'bve')
                assert {'vocals', 'bg_vocals'} <= stems.keys()
                for role in ('vocals', 'bg_vocals'):
                    write(folder / f'{role}.wav', stems[role].T)
            write_json(cache, {'identity': identity, 'run': separator.run_records[-1],
                'model_hashes': model_fingerprint(separator.model_file_dir, [model]),
                'outputs': {r: file_hash(folder / f'{r}.wav') for r in ('vocals', 'bg_vocals')}})
            print(f'Completed {source.parent.name}: {name}', flush=True)
    rng = random.Random(9431926)
    pairs, keys = [], []
    for source in sources:
        song = source.parent.name
        original = read(source)
        pool = {name: {role: read(OUT / 'predictions' / song / name / f'{role}.wav') for role in ('vocals', 'bg_vocals')} for name in BG_MODELS}
        pool['blend'] = {role: (pool['bve_v2'][role] + pool['karaoke'][role]) * .5 for role in ('vocals', 'bg_vocals')}
        for role, audio in pool['blend'].items():
            write(OUT / 'predictions' / song / 'blend' / f'{role}.wav', audio)
        for challenger in ('bve_v2', 'karaoke', 'blend'):
            number = len(pairs) + 1
            order = ['bve', challenger]
            rng.shuffle(order)
            pair = {'id': number, 'song': song, 'choices': []}
            key = {'id': number, 'song': song, 'labels': dict(zip('AB', order)), 'gains': {}}
            # Match leads downwards, apply the SAME gain to each associated backing
            # stem, then common headroom. Do not boost backing hiss for audition.
            target = min(rms(pool[n]['vocals']) for n in order)
            gains = {n: min(1., target / max(rms(pool[n]['vocals']), 1e-12)) for n in order}
            headroom = min(1., .98 / max(1e-12, max(float(abs(pool[n][r]).max()) * gains[n] for n in order for r in ('vocals', 'bg_vocals'))))
            for label, name in zip('AB', order):
                choice = {'label': label}
                gain = gains[name] * headroom
                key['gains'][label] = gain
                for role in ('vocals', 'bg_vocals'):
                    path = OUT / 'blind' / f'{number:02d}-{label}-{role}.wav'
                    write(path, pool[name][role] * gain)
                    choice[role] = path.relative_to(OUT).as_posix()
                pair['choices'].append(choice)
            path = OUT / 'blind' / f'{number:02d}-full-vocals.wav'
            write(path, original * min(1., .98 / max(float(abs(original).max()), 1e-12)))
            pair['original'] = path.relative_to(OUT).as_posix()
            pairs.append(pair)
            keys.append(key)
    write_json(OUT / 'key.json', keys)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Backing vocals — blind comparison</title>
<style>body{max-width:1000px;margin:32px auto;padding:20px;background:#141923;color:#e5eaf2;font:16px system-ui}section{background:#202838;padding:20px;margin:24px 0;border-radius:12px}.players{display:grid;grid-template-columns:1fr 1fr;gap:22px}audio{width:100%}label{display:block;margin:14px 0}select,textarea,button{font:inherit;padding:9px}textarea{width:95%}@media(max-width:650px){.players{grid-template-columns:1fr}}</style>
<h1>Backing vocals — A/B</h1><p>Pick the better separation: a clear lead with backing vocals removed, while keeping useful harmonies in the backing stem. Listen for missing lead words and lead leaking into the backing track.</p><p>18 pairs across six songs. Lead levels are matched downward; backing stems keep the same gain as their lead. Every approach starts with the cleaned hybrid vocals you selected.</p><button onclick="save()">Export scores</button><span id="status"></span><div id="pairs"></div>
<script>const pairs=DATA;const storage='audiolab-background-v1';let scores=JSON.parse(localStorage.getItem(storage)||'{}');function update(id,key,value){(scores[id]??={})[key]=value;localStorage.setItem(storage,JSON.stringify(scores));document.querySelector('#status').textContent=' Saved'}
function audio(parent,title,path){const p=document.createElement('p');p.textContent=title;parent.append(p);const a=document.createElement('audio');a.controls=true;a.preload='none';a.src=path;parent.append(a)}
for(const p of pairs){const s=document.createElement('section');const h=document.createElement('h2');h.textContent=p.id+'. '+p.song;s.append(h);const grid=document.createElement('div');grid.className='players';for(const c of p.choices){const box=document.createElement('div');audio(box,c.label+' — Lead vocals',c.vocals);audio(box,c.label+' — Backing vocals',c.bg_vocals);grid.append(box)}s.append(grid);const d=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Full vocals reference';d.append(summary);audio(d,'Before backing-vocal separation',p.original);s.append(d);const l=document.createElement('label');l.append(document.createTextNode('Winner '));const select=document.createElement('select');['Unrated','A','B','Tie'].forEach(o=>select.add(new Option(o,o)));select.value=scores[p.id]?.winner||'Unrated';select.onchange=()=>update(p.id,'winner',select.value);l.append(select);s.append(l);const notes=document.createElement('textarea');notes.placeholder='Notes';notes.value=scores[p.id]?.notes||'';notes.oninput=()=>update(p.id,'notes',notes.value);s.append(notes);document.querySelector('#pairs').append(s)}
function save(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:1,comparison:'background-v1',scores},null,2)],{type:'application/json'}));a.download='background-listening-scores.json';a.click();URL.revokeObjectURL(a.href)}
</script></html>'''.replace('=DATA;', '=' + json.dumps(pairs).replace('</', '<\\/') + ';')
    (OUT / 'listening.html').write_text(page, encoding='utf-8')
    (OUT / 'REPORT.md').write_text('''# Backing-vocal comparison

18 A/B comparisons: existing BVE versus BVE v2, Frazer–Becruily BS-RoFormer karaoke, and equal waveform averaging of the latter two (both lead and backing outputs). One pass on the exact cached listener-approved cleaned hybrid vocals; no recursive subtraction, cleanup or hidden stems. The blend is a hypothesis, not an assumed winner. Full vocals remain recoverable.

Sources: [upstream registry](https://github.com/nomadkaraoke/python-audio-separator/blob/v0.47.0/audio_separator/models.json), [karaoke author repository](https://huggingface.co/becruily/bs-roformer-karaoke). Registered BVE v2 means checkpoint UVR-BVE-4B_SN-44100-2.pth here, not a claim of equivalence to a hosted service. The karaoke author repository supplies no model card; no extra license grant is inferred.

Each prediction stores input/output and model/config hashes plus effective inference settings. Leads are RMS-matched downward; associated backing outputs receive the same gain, and each pair has common peak headroom. Backing stems are not independently normalized. Original-gain outputs are preserved in predictions/. Finite/rate/shape and lossless float roundtrip checks run on outputs. No reference-stem SDR or quality winner is claimed.

[Listen and score](listening.html). Only winner and notes are requested. Key is saved separately in key.json. Current BVE remains the background-vocal default until these comparisons are reviewed.
''', encoding='utf-8')
    print(OUT / 'listening.html', flush=True)


if __name__ == '__main__':
    main()
