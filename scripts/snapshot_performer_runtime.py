"""Archive the exact local implementation and foundation bundle manifest."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def main():
    destination=ROOT/'runtime-snapshots'/sys.argv[1]
    if destination.exists():raise ValueError('Runtime snapshot already exists')
    sources=list((REPO/'modules/rvc_v3/performer').glob('*.py'))
    sources += [REPO/p for p in ('scripts/performer_worker.py','scripts/performer_guide_worker.py',
        'scripts/rvc_backend_worker.py','scripts/benchmark_performer_v3.py','scripts/listen_performer_v3.py',
        'scripts/rerender_performer_timing.py','scripts/score_performer_words.py','scripts/prepare_performer_study.py',
        'modules/rvc/infer/modules/vc/pipeline.py','modules/rvc/infer/lib/infer_pack/models.py',
        'modules/rvc/pitch_extraction.py','modules/rvc/infer/lib/rmvpe.py',
        'modules/rvc/configs/config.py','requirements-performer-guide.txt','wrappers/clone.py')]
    records=[]
    for source in sources:
        output=destination/source.relative_to(REPO);output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,output);records.append(dict(source=identity(source),archive=identity(output)))
    model=Path('D:/models/AudioLab-performer/CosyVoice3')
    weights=[identity(p) for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts and p.suffix not in ('.png','.jpg','.md')]
    write_json(destination/'manifest.json',dict(schema=1,files=records,model_files=weights,
        repository_revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        guide_repository_revision=subprocess.check_output(['git','-C','E:/dev/AudioLab-experiments/CosyVoice','rev-parse','HEAD'],text=True).strip(),
        environment=identity(ROOT/'environment.txt'),note='Dirty working-tree files are archived explicitly; git HEAD alone is not sufficient provenance.'))
    print(destination)


if __name__=='__main__':main()
