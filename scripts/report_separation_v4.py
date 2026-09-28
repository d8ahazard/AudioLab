"""Build a local, blind, level-matched listening comparison and honest status report."""
import html
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] / "outputs/separation_v4_validation"


def main():
    corpus = json.loads((ROOT / "corpus.json").read_text())
    index = json.loads((ROOT / "fusion-index.json").read_text())
    trio = ["bs_roformer_vocals_resurrection_unwa.ckpt", "melband_roformer_big_beta7.ckpt", "mel_band_roformer_instrumental_becruily.ckpt"]
    chosen = [r for r in index if r["models"] == trio and r["algorithm"] == "avg_wave"]
    pairs = []
    rng = random.Random(24092026)
    listening = ROOT / "listening"
    listening.mkdir(exist_ok=True)
    for number, candidate in enumerate(chosen):
        role = candidate["role"]
        baseline_dir = ROOT / "baseline-0.41.1-v2" / candidate["song"]
        baseline = next(baseline_dir.glob(f"*__({role.title()}).wav"))
        paths = [baseline, Path(candidate["path"])]
        rng.shuffle(paths)
        arrays = [sf.read(p, dtype="float32", always_2d=True)[0] for p in paths]
        rms = [float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) for a in arrays]
        target_rms = min(rms)
        pair = {"id": number + 1, "song": candidate["song"], "role": role, "files": [], "key": []}
        for letter, path, arr, level in zip("AB", paths, arrays, rms):
            # Match downward only, for audition copies; original stems retain gain.
            gain = min(1., target_rms / max(level, 1e-12), .99 / max(float(np.abs(arr).max()), 1e-12))
            out = listening / f"{number + 1:02d}-{letter}.wav"
            sf.write(out, arr * gain, 44100, subtype="FLOAT")
            pair["files"].append(out.relative_to(ROOT).as_posix())
            pair["key"].append({"label": letter, "source": str(path), "gain": gain, "candidate": path != baseline})
        pairs.append(pair)
    (ROOT / "listening-key.json").write_text(json.dumps(pairs, indent=2))
    public = [{k: v for k, v in p.items() if k != "key"} for p in pairs]
    exploration = [{**r, "path": Path(r["path"]).relative_to(ROOT).as_posix()} for r in index]
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>AudioLab V4 listening comparison</title>
<style>body{max-width:1000px;margin:40px auto;padding:0 24px;background:#141923;color:#e5eaf2;font:16px system-ui}h1{font-size:28px}section{background:#202838;padding:22px;border-radius:12px;margin:20px 0}audio{width:100%}label{display:block;margin:12px 0}select,input,textarea,button{font:inherit;padding:8px;border-radius:6px}button{cursor:pointer}small{color:#b6c7dd}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}textarea{width:90%}</style>
<h1>AudioLab V4 — blind listening comparison</h1><p>Compare A and B for each stem. These are fixed excerpts from your recent songs, matched downward to the same RMS. The experimental candidate has not been promoted.</p>
<p>Listen for bleed, missing notes or harmonies, vocal clarity, and watery/buzzy artifacts. A quieter track is not automatically better. Scores stay in this browser until exported.</p>
<button onclick="save()">Export listening scores</button>
<details><summary>Explore all candidate recipes (not blinded, original gain)</summary><p>These controls select vocals and instrumentals independently. Prefer the blind comparisons below when judging the initial candidate against V2.</p><div id="explorer"></div><audio id="explorerAudio" controls preload="none"></audio></details>
<div id="pairs"></div>
<script>const pairs=DATA;const storeKey='audiolab-v4-listening-v1';let scores=JSON.parse(localStorage.getItem(storeKey)||'{}');
function update(id,key,value){(scores[id]??={})[key]=value;localStorage.setItem(storeKey,JSON.stringify(scores))}
for(const p of pairs){const section=document.createElement('section');const h=document.createElement('h2');h.textContent=p.id+'. '+p.song+' / '+p.role;section.append(h);const players=document.createElement('div');players.className='players';p.files.forEach((file,i)=>{const box=document.createElement('div');box.append(document.createTextNode('AB'[i]));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=file;box.append(audio);players.append(box)});section.append(players);
for(const [key,title,options] of [['winner','Preferred result',['Unrated','A','B','Tie']],['regression','Severe regression in either result?',['Unrated','Neither','A','B']],['bleed','Less bleed',['Unrated','A','B','Tie']],['detail','Better musical detail / harmonies',['Unrated','A','B','Tie']],['artifacts','Fewer artifacts',['Unrated','A','B','Tie']]]){const label=document.createElement('label');label.append(document.createTextNode(title+' '));const select=document.createElement('select');for(const option of options){select.add(new Option(option,option))}select.value=scores[p.id]?.[key]||'Unrated';select.onchange=()=>update(p.id,key,select.value);label.append(select);section.append(label)}
const notes=document.createElement('textarea');notes.placeholder='Notes and timestamps';notes.value=scores[p.id]?.notes||'';notes.oninput=()=>update(p.id,'notes',notes.value);section.append(notes);document.querySelector('#pairs').append(section)}
function save(){const blob=new Blob([JSON.stringify({schema:1,comparison:'v4-initial-vs-0411-v2',scores},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v4-listening-scores.json';a.click();URL.revokeObjectURL(a.href)}
const recipes=EXPLORER_DATA;const controls={};for(const key of ['song','role','models','algorithm']){const label=document.createElement('label');label.append(document.createTextNode(key+' '));const select=document.createElement('select');const values=[...new Set(recipes.map(r=>key==='models'?r.models.join(' + '):r[key]))];for(const value of values)select.add(new Option(value,value));select.onchange=()=>{const row=recipes.find(r=>Object.entries(controls).every(([k,s])=>(k==='models'?r.models.join(' + '):r[k])===s.value));document.querySelector('#explorerAudio').src=row?.path||''};controls[key]=select;label.append(select);document.querySelector('#explorer').append(label)}controls.algorithm.onchange();
</script></html>'''.replace('EXPLORER_DATA', json.dumps(exploration).replace('</', '<\\/')).replace('=DATA;', '=' + json.dumps(public).replace('</', '<\\/') + ';')
    (ROOT / "listening.html").write_text(document, encoding="utf-8")
    metrics = []
    for p in ROOT.glob("baseline-*/*/complete.json"):
        data = json.loads(p.read_text())
        metrics.append({"run": p.parent.parent.name, "song": p.parent.name, "seconds": data["seconds"], "peak_vram_bytes": data["peak_vram_bytes"]})
    (ROOT / "runtime-summary.json").write_text(json.dumps(metrics, indent=2))
    completed_predictions = list((ROOT / "predictions/balanced").glob("*/*/complete.json"))
    full = list((ROOT / "full-candidate").glob("*/complete.json"))
    specialists = list((ROOT / "specialists").glob("*/*/complete.json"))
    operational = [p.parent.name for p in (ROOT / "operational").glob("*/complete.json")]
    listening_status = (
        "Human listening scores have been submitted. See [decoded listening results](LISTENING_RESULTS.md) "
        "for preferences, regression flags, and unresolved entries; no recipe has been promoted."
        if (ROOT / "LISTENING_RESULTS.md").exists()
        else "No human listening scores have been submitted, so no quality winner is claimed."
    )
    report = f'''# AudioLab V4 validation report

Status: **experimental, not promoted**. {listening_status}

## Outputs

- {len(corpus)} recent source projects; existing project outputs were preserved.
- {len(completed_predictions)} cached core predictions: seven models across six fixed 24-second excerpt montages.
- {len(index)} fusion stem files: nine three-model recipes, five algorithms, two independently selectable stems, six songs.
- {len(specialists)} specialist comparisons completed; {len(full)} full-song candidate runs completed.
- [Blind listening comparison](listening.html), [recipe index](fusion-index.json), [runtime measurements](runtime-summary.json).
- `corpus.json` records source paths, excerpt offsets, and saved baseline metadata.
- Full-length output is under `full-candidate/`; its initial recipe remains experimental.
- Completed real-inference smoke scenarios: {', '.join(sorted(operational))}.
- Targeted automated tests and logs: `unit-tests-working-env.log`, `operational.log`.

## Installed runtime

The working environment imports AudioSeparator 0.47.0 and retains Torch 2.7.1+cu128. The verified runtime overlay selects ONNX Weekly 1.20.0.dev20251005, ONNX Runtime GPU 1.22.0, protobuf 4.25.8, and TensorBoardX 2.6.5. CUDA inference was exercised. Windows locked DLLs in the already-running app; the interrupted base-file replacement was repaired and the new runtime was staged without stopping that session. Restart AudioLab to activate it there. `runtime-overlay.log` records the installation; `environment-before.txt` preserves the original inventory.

Global `pip check` still reports pre-existing TTS/training dependency conflicts. TensorBoardX's protobuf incompatibility introduced by this upgrade was repaired, and TensorBoardX/AudioTools imports were checked. This report does not claim a clean dependency resolution for every AudioLab subsystem.

## Interpretation and limits

AudioSeparator 0.41.1/V2 and 0.47.0/V2 baselines use a preserved copy of the old AudioLab pipeline. Only eager downloads of unused models were bypassed. Candidate runs use gain-preserving float audio and author configs. No reference SDR is reported for commercial mixes without ground-truth stems. Runtime numbers are observed wall time; some jobs overlapped and these are not controlled speed rankings.

The old V3 pipeline with 0.47.0 exposed an upstream MDX spectral-inversion shape error. The new adapter explicitly uses waveform inversion for MDX models; the failure is retained in `baseline-v3.log`. Legacy V4 is benchmarked separately from the new V4 recipe.

Smart Stems uses conservative activity heuristics, not semantic recognition. Quiet tonal content, brief notes, transients, and uncertain results are retained; hidden files remain recoverable. A false positive on the controlled sparse-content tests blocks noise-only hiding.

## Promotion gate

Use the listening page to export scores. Require a majority of six songs to prefer V4 with no severe regression in either primary stem, plus backing-vocal and sparse-instrument checks. Ties do not count as wins. Until this gate is met, V2 remains the application default and V4 retains its experimental label. Alternative recipes and specialists remain available for comparison; published SDR figures are not used as fusion weights.
'''
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
