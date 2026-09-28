"""One isolated conversion job; JSON request/response, no UI imports for external models."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import hashlib
import subprocess
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def run(request):
    if request['backend']=='v3' and Path('D:/AI-outputs/AudioLab/rvc_quality_validation/v3/PAUSED').exists():
        raise RuntimeError('V3 is paused at the user request.')
    import numpy as np
    import torch
    import soundfile as sf
    torch.set_num_threads(4)
    seed = int(request.get('seed', 20260924))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    backend = request['backend']
    mode = request.get('mode', 'preserve')
    if mode not in {'preserve','target_style'}:
        raise ValueError(f'Unknown conversion mode: {mode}')
    if mode != 'preserve' and backend != 'vevo2':
        raise ValueError(f'{backend} does not support target-style conversion')
    if request.get('lyrics') and backend not in {'vevo2', 'v3'}:
        raise ValueError(f'{backend} does not support lyric conditioning')
    folder = Path(request['output_dir']).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if backend == 'rvc_v2':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from modules.rvc.configs.config import Config
        from modules.rvc.infer.modules.vc.pipeline import VC
        import modules.rvc.infer.modules.vc.pipeline as vc_module
        import faiss
        import logging
        faiss.omp_set_num_threads(4)
        logging.getLogger().setLevel(logging.WARNING)
        vc_module.output_path = str(folder)
        vc = VC(Config(), True)
        vc.get_vc(request['checkpoint'])
        # Never rely on substring-based index discovery for a benchmark.
        vc.index = request['index']
        paths = vc.vc_multi(model=request['checkpoint'], sid=0, paths=[request['source']],
            f0_up_key=request.get('pitch_shift',0), f0_method=request.get('pitch_method','rmvpe+'),
            index_rate=request.get('index_rate',1.), filter_radius=3, rms_mix_rate=.9,
            protect=request.get('protect',.2), merge_type='median', crepe_hop_length=160,
            f0_autotune=False, rmvpe_onnx=False, clone_stereo=False, pitch_correction=False,
            pitch_correction_humanize=.95, project_dir=str(folder), callback=None,
            use_model_warmup=False, warmup_duration=5.)
        if len(paths) != 1: raise RuntimeError('Expected one RVC output')
        output = Path(paths[0])
    elif backend == 'v3':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from modules.rvc_v3.io.checkpoint_io import load_inference_model
        from modules.rvc_v3.configs.v3_config import RVCV3Config
        from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
        _,_,metadata=load_inference_model(request['checkpoint'],device='cpu')
        os.environ['AUDIOCLONE_V3_TEXT_STRENGTH']=str(request.get('text_strength',0.))
        os.environ['AUDIOCLONE_V3_NOISE_SCALE']=str(request.get('noise_scale',.66666))
        pipeline=RVCV3Pipeline(request['checkpoint'],RVCV3Config.from_dict(metadata['config']),device='cuda')
        if request.get('index'): pipeline.load_retrieval_index(request['index'])
        output=folder/'converted.wav'
        pipeline.convert(request['source'],lyrics=request.get('lyrics'),output_path=str(output),
                         index_rate=request.get('index_rate',.5),pitch_shift=request.get('pitch_shift',0))
    elif backend in {'seed_vc', 'yingmusic'}:
        # Import PyAV's submodule before torchvision's optional video probe.
        import av.video
        repo = Path(request['repository']).resolve()
        os.chdir(repo)
        sys.path.insert(0,str(repo))
        if backend == 'seed_vc':
            import inference
            args = argparse.Namespace(source=request['source'], target=request['reference'],
                output=str(folder), diffusion_steps=request.get('steps',50), length_adjust=1.,
                inference_cfg_rate=.7, f0_condition=True, auto_f0_adjust=False,
                semi_tone_shift=request.get('pitch_shift',0), checkpoint=request.get('checkpoint'),
                config=request.get('config'), fp16=True)
            inference.main(args)
        else:
            import my_inference
            args = argparse.Namespace(source=request['source'], target=request['reference'],
                accompany=request.get('accompaniment'), output=str(folder), expname='render',
                diffusion_steps=request.get('steps',50), checkpoint=request['checkpoint'],
                config=request['config'], cuda=torch.device('cuda:0'), fp16=True,
                length_adjust=1., inference_cfg_rate=.7, f0_condition=True,
                semi_tone_shift=request.get('pitch_shift',0))
            models = my_inference.load_models_api(args, device=args.cuda)
            my_inference.run_inference(args, models, device=args.cuda)
        outputs = list(folder.rglob('*.wav'))
        if len(outputs) != 1: raise RuntimeError(f'Expected one output, found {outputs}')
        output = outputs[0]
    elif backend == 'vevo2':
        repo = Path(request['repository']).resolve()
        os.chdir(repo); sys.path.insert(0, str(repo))
        import whisper
        original_load = whisper.load_model
        def load_whisper(name, *args, **kwargs):
            kwargs.setdefault('download_root', 'D:/models/AudioLab-svc/whisper')
            return original_load(name, *args, **kwargs)
        whisper.load_model = load_whisper
        from models.svc.vevo2 import vevo2_utils
        # Portable SDPA/eager path; do not require a Blackwell-compatible flash-attn build.
        vevo2_utils.supported_flash_attn = False
        from models.svc.vevo2 import infer_vevo2_ar
        from huggingface_hub import snapshot_download
        infer_vevo2_ar.snapshot_download = lambda **kw: snapshot_download(**kw,
            ignore_patterns=['**/optimizer*','**/scheduler*','**/random_states*'],max_workers=2)
        pipeline = infer_vevo2_ar.load_inference_pipeline()
        if mode == 'target_style':
            lyrics = request.get('lyrics')
            reference_text = request.get('reference_text')
            if lyrics is None or reference_text is None:
                raise ValueError('Target-style Vevo2 requires source and reference transcripts')
            audio = pipeline.inference_ar_and_fm(target_text=lyrics,
                prosody_wav_path=request['source'], style_ref_wav_path=request['reference'],
                style_ref_wav_text=reference_text, timbre_ref_wav_path=request['reference'],
                use_prosody_code=True, use_pitch_shift=False,
                target_duration=sf.info(request['source']).duration,
                flow_matching_steps=request.get('steps',32))
        else:
            audio = pipeline.inference_fm(src_wav_path=request['source'],
                timbre_ref_wav_path=request['reference'], src_wav_text=request.get('lyrics') or '',
                timbre_ref_wav_text=request.get('reference_text') or '',
                use_pitch_shift=False,flow_matching_steps=request.get('steps',32))
        output=folder/'converted.wav'
        sf.write(output,audio.detach().cpu().squeeze().numpy(),24000,subtype='FLOAT')
    else:
        raise ValueError(f'Unknown or unconfigured backend: {backend}')
    audio, rate = sf.read(output, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all(): raise ValueError('Invalid audio output')
    def identity(path):
        digest=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
        return dict(path=str(path),sha256=digest.hexdigest())
    inputs={key:identity(request[key]) for key in ('source','checkpoint','index','reference','config')
            if request.get(key) and Path(request[key]).is_file()}
    if backend=='v3':
        inputs['metadata']=identity(Path(request['checkpoint']).with_name(Path(request['checkpoint']).stem+'_config.json'))
    bundle_manifest=Path('D:/AI-outputs/AudioLab/rvc_quality_validation/vevo-models.json')
    if backend=='vevo2' and bundle_manifest.exists(): inputs['model_bundle']=identity(bundle_manifest)
    repo=request.get('repository',str(Path(__file__).resolve().parents[1]))
    revision=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
    return dict(request=request, inputs=inputs, repository_revision=revision,
                worker=identity(__file__), torch_version=torch.__version__,
                output=str(output), output_identity=identity(output),sample_rate=rate,
                duration=len(audio)/rate, seconds=time.perf_counter()-start,
                peak=float(np.abs(audio).max()), clipped_fraction=float(np.mean(np.abs(audio)>=1)),
                peak_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                effective_settings=pipeline.last_convert_debug if backend=='v3' else None,
                timing_note='Wall time includes loading; other study jobs may share the GPU.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    from filelock import FileLock
    lock=Path(os.environ.get('AUDIOLAB_CONVERSION_LOCK',
        'D:/AI-outputs/AudioLab/rvc_quality_validation/.conversion.lock'))
    lock.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(lock)):
        result=run(request)
        target=Path(request['output_dir'])/'result.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2),encoding='utf-8')
        temporary.replace(target)
    print(target)
