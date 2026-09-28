"""Independent ASR evidence for disputed words; never auto-approve a manual review."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modules.rvc_v3.performer.contracts import identity,write_json
ROOT=Path('D:/AI-outputs/AudioLab/performer_v3_validation')


def main():
    import torch
    import whisper
    from filelock import FileLock
    torch.set_num_threads(4)
    with FileLock('D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'):
        model=whisper.load_model('medium',device='cuda',download_root='D:/models/AudioLab-svc/whisper')
        for case in sys.argv[2:]:
            folder=ROOT/'rounds'/sys.argv[1]/case/'performer'
            result=json.loads((folder/'result.json').read_text())
            observations=[]
            for role,path in [('source',result['source']['path']),('output',result['output'])]:
                transcription=model.transcribe(path,language='en',temperature=0,condition_on_previous_text=False,fp16=True)
                observations.append(dict(role=role,audio=identity(path),text=transcription['text']))
            write_json(folder/'independent-word-audit.json',dict(model=identity('D:/models/AudioLab-svc/whisper/medium.pt'),
                reviewed=False,observations=observations,note='Independent automatic evidence only; disagreements still require listening.'))
            print(case,flush=True)


if __name__=='__main__':main()
