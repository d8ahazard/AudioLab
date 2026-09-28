"""Fit on the first two excerpts; evaluate/replay on the untouched third excerpt."""
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.reverb_ir import fit_effect_ir, convolve_effect, save_capture, REVISION
from modules.separator.stem_manifest import write_json, file_hash

ROOT = REPO / 'outputs/separation_v4_validation/royals-reverb'


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    original, sr = sf.read(manifest['source'], always_2d=True)
    blocks, results = [], []
    for model in manifest['models']:
        slug = model['slug']
        folder = ROOT/'predictions'/slug
        dry, rate = sf.read(folder/'dry.wav', always_2d=True)
        assert rate == sr
        removed = original - dry
        target = folder/'stereo-ir.json'
        start = time.perf_counter()
        if target.exists():
            capture = json.loads(target.read_text())
            assert capture['revision'] == REVISION and capture['source_hash'] == file_hash(folder/'dry.wav')
        else:
            capture = fit_effect_ir(dry, removed, sr, segment_frames=8*sr)
            capture['source_hash'] = file_hash(folder/'dry.wav')
            save_capture(target, capture)
        impulse = np.array(capture['impulse_response'])
        # Every 8-second block has its own real-song context; don't smear reverb across montage joins.
        effect = np.concatenate([convolve_effect(dry[s:s+sr*8], impulse) for s in range(0,len(dry),sr*8)])
        replay = dry + effect
        # A shared downward gain for replay + its original reference, never clipping.
        gain = min(1., .98/max(float(abs(replay).max()),float(abs(original).max()),1e-12))
        for role, audio in [('ir-replay',replay),('ir-reference',original)]:
            path = ROOT/'audio'/f'{slug}-{role}.wav'
            sf.write(path, audio*gain, sr, subtype='FLOAT')
            check, rate = sf.read(path,always_2d=True)
            assert rate == sr and check.shape == original.shape and np.isfinite(check).all()
        valid = capture['valid_for_restore']
        d = capture['diagnostics']
        result = {'model':model['name'],'slug':slug,'valid_for_restore':valid,'diagnostics':d,
                  'rejection_reasons':capture['rejection_reasons'],'preview_gain':gain,'seconds':time.perf_counter()-start}
        results.append(result)
        status = 'Passes automatic restore check' if valid else 'Preview only — not approved for automatic restore'
        blocks.append(f'<section><h2>{html.escape(model["name"])}</h2><p><strong>{status}</strong></p>'
            f'<p>Held-out removed-signal error reduction: {d["holdout_effect_error_reduction_percent"]:.1f}%. '
            f'Trained on 0–16 seconds; evaluation uses 17.5–24 seconds, allowing 1.5 seconds of convolution context.</p>'
            f'<p>{html.escape("; ".join(capture["rejection_reasons"]))}</p><div class="players">'
            f'<div><h3>Original lead reference</h3><audio controls preload="none" src="audio/{slug}-ir-reference.wav"></audio></div>'
            f'<div><h3>Dry vocal + newly estimated IR</h3><audio controls preload="none" src="audio/{slug}-ir-replay.wav"></audio></div></div></section>')
        print(slug, status, round(d['holdout_effect_error_reduction_percent'],1), flush=True)
    write_json(ROOT/'ir-evaluation.json', {'revision':REVISION,'results':results,
        'note':'Removed-signal reconstruction diagnostics, not reference SDR or proof of true plugin recovery. Replays include fitted and held-out excerpts; judge the last excerpt for generalization.'})
    page = (ROOT/'index.html').read_text(encoding='utf-8')
    marker = '<!-- IR-EVALUATION -->'
    # Idempotent insertion keeps the named model page and local notes intact.
    if marker in page:
        page = page[:page.index(marker)] + '</html>'
    section = marker + '<h1>New IR capture / reapply test</h1><p>These are diagnostic previews, using each model’s estimated dry vocal. The first two excerpts fit the IR; the last excerpt is held out. Check the last 6.5 seconds for transfer quality. A failed preview is never enabled for automatic restoration. No cloned voice was used in this test.</p>' + '\n'.join(blocks)
    page = page.replace('</html>', section+'</html>')
    (ROOT/'index.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    main()
