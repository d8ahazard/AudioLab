"""Honest reconstruction of the existing 29 optional orchestral Mega outputs."""
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.separator.instrument_policy import MEGA_EXTRAS, duplicate_of
from modules.separator.stem_manifest import file_hash, write_json


def main():
    base = ROOT/'outputs/separation_v4_validation/orchestral-mega'
    out = base/'reconstruction'
    out.mkdir(exist_ok=True)
    metadata = json.loads((base/'manifest.json').read_text())
    by_role = {t['role']:t for t in metadata['tracks']}
    def read(role):
        item = by_role[role]
        path = base/item['path']
        assert file_hash(path) == item['sha256'], path
        a,sr = sf.read(path,dtype='float32',always_2d=True)
        assert sr == 44100 and a.shape == (72*44100,2) and np.isfinite(a).all()
        return a
    original = read('reference')
    # All source players used the same gain. Preserve their relative levels.
    reconstruction = np.zeros_like(original,dtype='float64')
    used, duplicates, pool = [], {}, {}
    for role in MEGA_EXTRAS:
        a = read(role)
        duplicate = duplicate_of(a.T, pool)
        if duplicate:
            duplicates[role] = duplicate
            continue
        pool[role] = a.T
        reconstruction += a
        used.append(role)
    difference = original.astype('float64') - reconstruction
    peak = max(float(abs(a).max()) for a in [original,reconstruction,difference])
    gain = min(1., .98/max(peak,1e-12))
    outputs = {}
    for name,a in [('original',original),('reconstructed',reconstruction),('difference',difference)]:
        path = out/(name+'.wav')
        sf.write(path,a*gain,44100,subtype='FLOAT')
        check,sr = sf.read(path,dtype='float32',always_2d=True)
        assert check.shape == original.shape and np.isfinite(check).all() and abs(check).max() <= .98001
        outputs[name] = {'sha256':file_hash(path),'rms_before_playback_gain':float(np.sqrt(np.mean(a.astype('float64')**2)))}
    write_json(out/'manifest.json',{'source_manifest_sha256':file_hash(base/'manifest.json'),
        'roles_summed':used,'duplicates_excluded':duplicates,'common_playback_gain':gain,
        'source_common_gain':metadata['common_gain'],'starts':metadata['starts'],
        'outputs':outputs,'seconds':72,'scope':'Existing Mega extras only; no blended core predictions available for this recording',
        'limitations':'Different Mega heads may still share musical content. No residual fill, denoising, per-track normalization or forced mixture consistency.'})
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Original vs reconstructed orchestra</title>
<style>body{background:#101820;color:#edf4fa;font:17px system-ui;max-width:900px;margin:45px auto;padding:20px}p{line-height:1.6;color:#bfd0df}section{background:#203242;padding:24px;margin:18px 0;border-radius:12px}audio{width:100%}button{padding:10px 16px;margin-right:8px}a{color:#8acfff}</style>
<h1>Original vs reconstructed orchestra</h1>
<p>Russian Easter Festival Overture · Musopen recording. Six excerpts, 72 seconds total.</p>
<p>The reconstruction sums the existing 29 optional Mega instrument estimates. This is an <b>extras-only reconstruction</b>: the blended core instruments were not generated for this recording. Some Mega estimates may still overlap. Nothing from the original was added to repair missing content.</p>
<p>Both versions use the same playback gain and sample alignment. Switch buttons continue from the current playback position.</p>
<button onclick="swap('original')">Hear original</button><button onclick="swap('reconstructed')">Hear reconstruction</button>
<section><h2>Original</h2><audio id="original" controls preload="metadata" src="original.wav"></audio></section>
<section><h2>Reconstructed from separated extras</h2><audio id="reconstructed" controls preload="metadata" src="reconstructed.wav"></audio></section>
<details><summary>Hear the difference signal</summary><p>Original minus reconstruction. This contains missing content and subtraction/overlap errors; it is not a clean additional stem.</p><audio id="difference" controls preload="none" src="difference.wav"></audio></details>
<p><a href="manifest.json">Exact stem list, gains and file hashes</a> · <a href="../extras.html">Individual stems</a></p>
<script>let active=null;const players=[...document.querySelectorAll('audio')];players.forEach(a=>a.onplay=()=>{players.forEach(b=>{if(b!==a)b.pause();});active=a;});function swap(id){let a=document.getElementById(id),t=active?active.currentTime:0;if(active)active.pause();a.currentTime=t;a.play();}</script></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({'summed':len(used),'duplicates':duplicates,'gain':gain,'page':str(out/'index.html')}))


if __name__=='__main__':main()
