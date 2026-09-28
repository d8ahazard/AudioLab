"""Summarize evidence without treating rendered audio as a passed research gate."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import write_json,identity
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def main():
    entries=[]
    for path in sorted((ROOT/'rounds').glob('*/*/*/result.json')):
        data=json.loads(path.read_text())
        entry=dict(round=path.parents[2].name,case=path.parents[1].name,mode=path.parent.name,
                   report=identity(path),quality_status=data.get('quality_status','existing_v2'),
                   output=data['output_identity'],sample_rate=data['sample_rate'],
                   seconds=data.get('wall_seconds'),duration=data.get('timeline_seconds',data.get('duration')))
        if 'guide_report' in data:
            guide=data['guide_report']
            entry.update(guide_seconds=guide['seconds'],guide_peak_torch_vram_bytes=guide['peak_torch_vram_bytes'],
                         guide_peak_process_ram_bytes=guide['process_memory'].get('peak_wset'),reference_reviewed=data['reference_reviewed'])
        entries.append(entry)
    failures=[]
    for path in sorted(ROOT.glob('**/failure.json')):
        failures.append(dict(path=str(path),**json.loads(path.read_text())))
    reviews=[dict(record=identity(path),**json.loads(path.read_text()))
             for path in sorted((ROOT/'rounds').glob('*/listening-review-*.json'))]
    failed_reviews=[r for r in reviews if r.get('gate_passed') is False]
    report=dict(schema=1,entries=entries,failures=failures,listening_reviews=reviews,
                conclusion='M1 render feasibility only. Listening and transcript/target-only review are required; no learned-delivery improvement established.')
    if failed_reviews:
        report['conclusion']='Tested M1 guide configuration failed human listening on identity/pronunciation. Diagnose guide, timing and renderer separately before further adaptation; learned performer delivery remains untested.'
    write_json(ROOT/'results-summary.json',report)
    write_json(ROOT/'milestones.json',dict(M0='partial: inventories/contracts available; target segmentation and corrected alignment pending',
        M1=('failed for reviewed guide configuration: identity/pronunciation; diagnosis required' if failed_reviews else 'render experiments recorded; listening gate pending'),M2='gated: no trained performer adapters or bridge',
        M3='edit-window API exists; natural lyric-edit quality unproven',M4='not evaluated',M5='not released'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
