"""Report review state and preserve reference provenance; do not approve segments."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import write_json,identity
from modules.rvc_v3.performer.data import coverage
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')
OLD=Path('D:/AI-outputs/AudioLab/rvc_quality_validation')


def main():
    reports={}
    refs=json.loads((OLD/'references.json').read_text())
    for path in (ROOT/'profiles').glob('*.json'):
        profile=json.loads(path.read_text());voice=profile['id']
        if voice in refs:
            source=refs[voice]['source'];start=refs[voice]['start']
        else:
            needle='Dear Mama' if voice=='tupac' else 'Faint'
            record=next(r for r in profile['recordings'] if needle in r['recording_id'])
            source=record['source'];start=40 if voice=='tupac' else None
        # Store added audit information separately so frozen profile hashes do not change.
        reports[voice]=dict(**coverage(profile),profile=identity(path),reference_source=source,
            reference_start=start,reference_reviewed=profile['reference_reviewed'],
            candidate_only=True,adaptation_permitted=False)
    write_json(ROOT/'coverage.json',reports)
    print(json.dumps({k:{'inventory_minutes':v['inventory_minutes'],'reviewed_minutes':v['reviewed_minutes']} for k,v in reports.items()},indent=2))


if __name__=='__main__':main()
