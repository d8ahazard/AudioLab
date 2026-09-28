"""Audit preservation and report completed artifacts without inferring quality winners."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT,fingerprint,write_json

manifest=json.loads((ROOT/'corpus.json').read_text())
protected=manifest['hux_originals']+list(manifest['models'].values())+[manifest['shine_checkpoint']]
protected += [x['source'] for x in manifest['corpus']]
protected += [x['old_lead'] for x in manifest['corpus'] if x.get('old_lead')]
originals=ROOT/'training/original/sources.json'
if originals.exists():protected += [x['source'] for x in json.loads(originals.read_text())]
changed=[x['path'] for x in protected if not Path(x['path']).exists() or fingerprint(x['path'])['sha256']!=x['sha256']]
if changed:raise RuntimeError(f'Original asset integrity mismatch: {changed}')
results=[]
for p in (ROOT/'runs').glob('*/*/result.json'):
    result=json.loads(p.read_text());results.append(dict(case=p.parent.parent.name,variant=p.parent.name,backend=result['request']['backend']))
state=dict(updated=datetime.now(timezone.utc).isoformat(),protected_files_verified=len(protected),hux_originals_unchanged=len(manifest['hux_originals']),
    separation_complete=[p.parent.name for p in (ROOT/'separation').glob('*/complete.json')],
    required_separations=len(manifest['corpus']),excerpt_renders=len(results),
    backend_counts=dict(Counter(x['backend'] for x in results)),
    full_song_renders=len(list((ROOT/'full_songs').glob('*/*/result.json'))),
    rvc_adaptations_complete=[p.parent.parent.name for p in (ROOT/'training').glob('*/run/complete.json')],
    seed_adaptations_complete=[p.parent.name for p in (ROOT/'seed_training').glob('*/complete.json')],
    v3_text_pilots_complete=[p.parent.name for p in (ROOT/'v3_text').glob('*/training.json')],
    stage_status={p.stem:json.loads(p.read_text()) for p in (ROOT/'stages').glob('*.json')},
    perceptual_verdict='Pending blinded listening; no model promoted',
    limitations=['Two voices and two test songs only','Passage vocal-technique labels need human review',
                 'Simple Man held out from adaptation, not historical RVC training','ASR lyrics unreviewed',
                 'Runtime includes loading and shared GPU activity; VRAM is PyTorch allocation peak'])
write_json(ROOT/'study_status.json',state)
print(json.dumps(state,indent=2))
