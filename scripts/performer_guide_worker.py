"""CosyVoice guide generation in its own environment, never the final RVC renderer."""
import json
import os
from pathlib import Path
import sys
import time
import subprocess
os.environ.setdefault('HF_HOME','D:/models/hf-cache')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def generate(model,text,request):
    prefix='You are a helpful assistant.<|endofprompt|>'
    if request.get('conditioning','transcript')=='audio_only':
        return model.inference_cross_lingual(prefix+text,request['reference'],stream=False,text_frontend=False)
    return model.inference_zero_shot(text,prefix+request['reference_text'],request['reference'],stream=False,text_frontend=False)


def main():
    req=json.loads(Path(sys.argv[1]).read_text());repo=Path(req.get('repository','E:/dev/AudioLab-experiments/CosyVoice'))
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'third_party/Matcha-TTS'))
    import av.video
    import torch
    import numpy as np
    import random
    import soundfile as sf
    torch.set_num_threads(4);torch.manual_seed(req['seed']);random.seed(req['seed']);np.random.seed(req['seed'])
    from cosyvoice.cli.cosyvoice import AutoModel
    from modules.rvc_v3.performer.contracts import identity,write_json
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model=AutoModel(model_dir=req.get('model','D:/models/AudioLab-performer/CosyVoice3'),fp16=False)
    outputs=[]
    for phrase in req['phrases']:
        parts=[x['tts_speech'].cpu() for x in generate(model,phrase['text'],req)]
        if not parts:raise RuntimeError('No guide audio generated')
        audio=torch.cat(parts,dim=1).squeeze().numpy()
        if not np.isfinite(audio).all():raise RuntimeError('Nonfinite guide')
        dest=Path(req['output_dir'])/(phrase['id']+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
        sf.write(dest,audio,model.sample_rate,subtype='FLOAT')
        outputs.append(dict(id=phrase['id'],audio=identity(dest),sample_rate=model.sample_rate,seconds=len(audio)/model.sample_rate))
    import psutil
    model_dir=Path(req.get('model','D:/models/AudioLab-performer/CosyVoice3'))
    write_json(Path(req['output_dir'])/'guides.json',dict(stage='unadapted_pretrained_guide',outputs=outputs,conditioning=req.get('conditioning','transcript'),
        repository_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        worker=identity(__file__),model_files=[identity(model_dir/p) for p in ('llm.pt','flow.pt','hift.pt','cosyvoice3.yaml')],
        peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),process_memory=psutil.Process().memory_info()._asdict(),seconds=time.perf_counter()-started))


if __name__=='__main__':main()
