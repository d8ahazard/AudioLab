"""Build cached V2-vocal/V4-instrumental experiments and a blind listening page."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1] / 'outputs/separation_v4_validation'
OUT = ROOT / 'hybrid-comparison'
SETTINGS = dict(nperseg=2048, noverlap=1536)


def clean_vocals(vocals, residual, instrumental):
    """Experimental shared-stereo mask; at most 3 dB suppression, no added audio.

    Suppress only bins where the instrumental dominates and the residual provides
    little support for V2. This is evidence of possible bleed, not a voice detector.
    """
    spectra = [signal.stft(x.T, **SETTINGS)[2] for x in (vocals, residual, instrumental)]
    v, r, i = spectra
    vm, rm, im = [np.sqrt(np.mean(abs(x) ** 2, axis=0)) for x in spectra]
    disagreement = np.clip(1 - rm / (vm + 1e-10), 0, 1)
    dominance = np.clip((im / (vm + 1e-10) - 2) / 4, 0, 1)
    evidence = uniform_filter(disagreement * dominance, size=(3, 5), mode='nearest')
    mask = 1 - (1 - 10 ** (-3 / 20)) * evidence
    _, audio = signal.istft(v * mask[None], **SETTINGS)
    return audio[:, :len(vocals)].T.astype(np.float32), {
        'minimum_mask': float(mask.min()), 'mean_mask': float(mask.mean()),
        'fraction_bins_attenuated_over_1db': float(np.mean(mask < 10 ** (-1 / 20))),
    }


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def read(path):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    assert rate == 44100 and audio.shape[1] == 2 and np.isfinite(audio).all(), path
    return audio


def write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 44100, subtype='FLOAT')
    assert np.array_equal(read(path), audio.astype(np.float32)), path


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def self_check():
    rng = np.random.default_rng(13)
    x = rng.normal(0, .01, (16384, 2)).astype(np.float32)
    unchanged, _ = clean_vocals(x, x, x * 4)
    assert np.max(abs(unchanged - x)) < 1e-7
    anti = np.column_stack([x[:, 0], -x[:, 0]])
    cleaned, stats = clean_vocals(anti, anti * 0, anti * 10)
    assert np.max(abs(cleaned[:, 0] + cleaned[:, 1])) < 1e-7
    assert stats['minimum_mask'] >= 10 ** (-3 / 20) - 1e-6
    assert .70 < rms(cleaned) / rms(anti) < .72
    silent, _ = clean_vocals(x * 0, x * 0, x * 0)
    assert not np.any(silent)


def main():
    self_check()
    key = json.loads((ROOT / 'listening-key.json').read_text())
    pairs, metrics, private_key = [], [], []
    rng = random.Random(73512026)
    for number, item in enumerate((p for p in key if p['role'] == 'instrumental'), 1):
        song = item['song']
        baseline = ROOT / 'baseline-0.41.1-v2' / song
        paths = {'source': ROOT / 'excerpts' / (song + '.wav'),
                 'v2_vocals': next(baseline.glob('*__(Vocals).wav')),
                 'v2_instrumental': next(baseline.glob('*__(Instrumental).wav')),
                 'v4_instrumental': Path(next(k['source'] for k in item['key'] if k['candidate']))}
        arrays = {name: read(path) for name, path in paths.items()}
        source, v2, v2i, inst = (arrays[k] for k in paths)
        assert all(x.shape == source.shape for x in arrays.values()), song
        residual = source - inst
        cleaned, stats = clean_vocals(v2, residual, inst)
        candidates = {'hybrid': v2, 'residual': residual, 'cleaned': cleaned}
        folder = OUT / 'audio' / song
        write(folder / 'instrumental.wav', inst)
        row = {'song': song, 'inputs': {k: fingerprint(p) for k, p in paths.items()},
               'frames': len(source), 'sample_rate': 44100,
               'source_channels': sf.info(paths['source']).channels,
               'mono_handling': 'duplicate mono channels without gain change',
               'cleanup': stats, 'candidates': {}}
        for name, vocal in candidates.items():
            recombined = vocal + inst
            error = recombined - source
            write(folder / f'{name}-vocals.wav', vocal)
            write(folder / f'{name}-recombined.wav', recombined)
            row['candidates'][name] = {
                'reconstruction_error_db_relative_to_mix': 20 * np.log10(max(rms(error), 1e-15) / max(rms(source), 1e-15)),
                'peak': float(abs(recombined).max()),
                'samples_over_full_scale': int(np.count_nonzero(abs(recombined) > 1)),
                'vocal_change_db_relative_to_v2': 20 * np.log10(max(rms(vocal-v2), 1e-15) / max(rms(v2), 1e-15)),
            }
        assert np.max(abs(residual + inst - source)) < 1e-6
        row['v2_reconstruction_error_db_relative_to_mix'] = 20 * np.log10(max(rms(v2 + v2i - source), 1e-15) / max(rms(source), 1e-15))
        metrics.append(row)
        order = list(candidates)
        rng.shuffle(order)
        target = min(rms(x) for x in candidates.values())
        gains = {name: min(1., target / max(rms(x), 1e-15)) for name, x in candidates.items()}
        # Common extra headroom preserves equal RMS even if one candidate has high peaks.
        headroom = min(1., .98 / max(float(abs(candidates[n]).max()) * gains[n] for n in order))
        mix_gain = min(1., .98 / max(float(abs(x).max()) for x in [source, v2+v2i] + [v+inst for v in candidates.values()]))
        public = {'id': number, 'song': song, 'choices': []}
        secret = {'id': number, 'song': song, 'labels': {}, 'vocal_gains': {}, 'mix_gain': mix_gain}
        for label, name in zip('ABC', order):
            vocal_path = OUT / 'blind' / f'{number:02d}-{label}-vocals.wav'
            mix_path = OUT / 'blind' / f'{number:02d}-{label}-mix.wav'
            gain = gains[name] * headroom
            write(vocal_path, candidates[name] * gain)
            write(mix_path, (candidates[name] + inst) * mix_gain)
            public['choices'].append({'label': label, 'vocal': vocal_path.relative_to(OUT).as_posix(), 'mix': mix_path.relative_to(OUT).as_posix()})
            secret['labels'][label] = name
            secret['vocal_gains'][label] = gain
        for label, audio in [('original', source), ('v2-pair', v2 + v2i)]:
            dest = OUT / 'blind' / f'{number:02d}-{label}.wav'
            write(dest, audio * mix_gain)
            public[label] = dest.relative_to(OUT).as_posix()
        pairs.append(public)
        private_key.append(secret)
    OUT.mkdir(exist_ok=True)
    (OUT / 'key.json').write_text(json.dumps(private_key, indent=2))
    (OUT / 'manifest.json').write_text(json.dumps({'revision': 'hybrid-experiment-1', 'settings': SETTINGS,
        'max_suppression_db': 3, 'checks': 'synthetic identity, suppression bound, stereo phase, silence, float WAV roundtrip, exact shapes/rates, residual reconstruction passed',
        'songs': metrics}, indent=2))
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Hybrid separation listening</title>
<style>body{max-width:1100px;margin:32px auto;padding:20px;background:#141923;color:#e5eaf2;font:16px system-ui}section{background:#202838;padding:20px;margin:24px 0;border-radius:12px}.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}audio{width:100%}label{display:block;margin:12px 0}select,textarea,button{font:inherit;padding:8px}textarea{width:95%}small{color:#bcc9db}@media(max-width:750px){.choices{grid-template-columns:1fr}}</style>
<h1>Hybrid separation — vocal comparison</h1><p>Six 24-second excerpt montages. Each song has three randomized vocal choices; one is untouched V2. All three use the same V4 instrumental. Vocals are RMS-matched downward. Judge vocal bleed, missing words and harmonies, and artifacts.</p>
<p>Combined mixes use one common gain per song, including the original and V2 reference pair. They are not individually loudness-matched: listen for overlap, gaps, and balance changes. Exact reconstruction alone does not mean clean separation. These are experimental candidates, not a promoted profile.</p>
<button onclick="save()">Export scores</button><p id="status"></p><div id="songs"></div>
<script>const pairs=DATA;const storage='audiolab-hybrid-experiment-1';let scores=JSON.parse(localStorage.getItem(storage)||'{}');
function update(id,key,value){(scores[id]??={})[key]=value;localStorage.setItem(storage,JSON.stringify(scores));document.querySelector('#status').textContent='Saved in this browser'}
function player(parent,title,path){const p=document.createElement('p');p.textContent=title;parent.append(p);const a=document.createElement('audio');a.controls=true;a.preload='none';a.src=path;parent.append(a)}
for(const p of pairs){const s=document.createElement('section');const h=document.createElement('h2');h.textContent=p.id+'. '+p.song;s.append(h);const grid=document.createElement('div');grid.className='choices';for(const c of p.choices){const box=document.createElement('div');player(box,c.label+' vocals',c.vocal);player(box,c.label+' + instrumental',c.mix);grid.append(box)}s.append(grid);const refs=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Original mix and V2 pair references';refs.append(summary);player(refs,'Original mix',p.original);player(refs,'V2 vocals + V2 instrumental',p['v2-pair']);s.append(refs);
for(const [key,title,options] of [['winner','Preferred vocals',['Unrated','A','B','C','Tie']],['bleed','Least instrumental bleed',['Unrated','A','B','C','Tie']],['detail','Best vocal detail / quiet harmonies',['Unrated','A','B','C','Tie']],['artifacts','Fewest artifacts',['Unrated','A','B','C','Tie']],['mix','Best combined mix',['Unrated','A','B','C','V2 reference','Tie']]]){const l=document.createElement('label');l.append(document.createTextNode(title+' '));const select=document.createElement('select');options.forEach(o=>select.add(new Option(o,o)));select.value=scores[p.id]?.[key]||'Unrated';select.onchange=()=>update(p.id,key,select.value);l.append(select);s.append(l)}
for(const c of p.choices){const l=document.createElement('label');l.append(document.createTextNode('Severe regression in '+c.label+' vocals? '));const select=document.createElement('select');['Unrated','No','Yes'].forEach(o=>select.add(new Option(o,o)));const key='regression_'+c.label;select.value=scores[p.id]?.[key]||'Unrated';select.onchange=()=>update(p.id,key,select.value);l.append(select);s.append(l)}
const notes=document.createElement('textarea');notes.placeholder='Missing words, harmonies, artifacts, timestamps';notes.value=scores[p.id]?.notes||'';notes.oninput=()=>update(p.id,'notes',notes.value);s.append(notes);document.querySelector('#songs').append(s)}
function save(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({schema:1,comparison:'hybrid-experiment-1',scores},null,2)],{type:'application/json'}));a.download='hybrid-listening-scores.json';a.click();URL.revokeObjectURL(a.href)}
</script></html>'''.replace('=DATA;', '=' + json.dumps(pairs).replace('</', '<\\/') + ';')
    (OUT / 'listening.html').write_text(document, encoding='utf-8')
    lines = ['# Hybrid separation experiment', '', 'Experimental; V2 remains the application default.', '',
        'Three candidates: untouched cached 0.41.1/V2 vocals with V4 instrumental; original mix minus V4 instrumental; V2 vocals with a residual-guided shared-stereo attenuation mask capped at 3 dB. No new inference or upward normalization. Existing source and prior outputs are untouched.', '',
        'The cleanup is a hypothesis, not guaranteed voice protection. Harmonics shared with instruments may still be attenuated. Stereo channels use the same mask, retain V2 spectral phase, and receive no residual addition. The baseline contains legacy gain behavior; no undocumented gain correction or time shifting was applied.', '',
        '[Blind listening page](listening.html). Vocals are matched to the lowest RMS; all combined mixes and references share one headroom gain per song. Original-gain files are in `audio/`. Candidate identities are deliberately omitted here; `key.json` preserves them for scoring.', '',
        'Shape/rate/finite checks and float WAV roundtrips passed for every output. Synthetic checks passed for identity when residual agrees, <=3 dB mask suppression, anti-phase stereo, and silence. These checks cannot establish improved perceptual quality.', '',
        'Reconstruction error below is relative RMS error, not reference-stem SDR or a separation-quality score. Residual reconstruction is exact by construction. The hybrid can overlap or omit content.', '',
        '| Song | V2 pair error dB | Hybrid error dB | Cleaned error dB | Residual error dB |', '|---|---:|---:|---:|---:|']
    for row in metrics:
        errors = [row['v2_reconstruction_error_db_relative_to_mix']] + [row['candidates'][n]['reconstruction_error_db_relative_to_mix'] for n in ('hybrid','cleaned','residual')]
        lines.append('| '+row['song']+' | '+' | '.join(f'{e:.1f}' for e in errors)+' |')
    lines += ['', 'Outputs are 44.1 kHz stereo with the exact cached excerpt length. The mono lucifer source is duplicated into stereo without gain change to match its cached predictions. `manifest.json` contains input hashes, measured peaks, over-full-scale sample counts, mask statistics, and settings. Float originals may exceed full scale; listening copies have shared peak headroom. No clipping or limiting was applied.', '',
        'Select a vocal candidate only after listening, including quiet harmonies and consonants; then validate full songs and backing-vocal specialists. The hybrid uses both pipelines and is not a three-model production recipe or a speed improvement.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT / 'listening.html')


if __name__ == '__main__':
    main()
