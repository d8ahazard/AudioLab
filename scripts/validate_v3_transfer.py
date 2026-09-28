"""Export both actual singers through the repaired V3 backbone and test waveform parity."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.benchmark_rvc_quality import ROOT, OUTPUTS, MODELS, write_json
from modules.rvc_v3.training.expand_weights import expand_v2_to_v3
from modules.rvc_v3.configs.v3_config import RVCV3Config
from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
from modules.rvc.infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from modules.rvc_v3.io.checkpoint_io import load_inference_model

torch.set_num_threads(4)
for voice,run,epoch,inference in [('huxlxy','huxlxy_v2',500,MODELS/'huxlxy_v2_v500.pth'),
                                ('shinedown','Shinedown666',75,ROOT/'models/Shinedown666_e75.pth')]:
    folder=ROOT/'v3'/voice
    checkpoint=folder/'f0G48k.safetensors'
    if not checkpoint.exists():
        expand_v2_to_v3(str(OUTPUTS/f'voices/{run}/saves/G_e{epoch}.pth'),
                        str(OUTPUTS/f'voices/{run}/saves/D_e{epoch}.pth'),str(folder),48000,True)
    _,_,metadata=load_inference_model(str(checkpoint),device='cpu')
    cfg=RVCV3Config.from_dict(metadata['config'])
    # Initialize generator alone for the contract check (no content extraction differences).
    shell=RVCV3Pipeline.__new__(RVCV3Pipeline);shell.config=cfg;shell.device='cpu'
    shell._init_generator()
    gen,_,_=load_inference_model(str(checkpoint),device='cpu')
    shell.generator.load_state_dict(gen,strict=True)
    exported=torch.load(inference,map_location='cpu',weights_only=True)
    baseline=SynthesizerTrnMs768NSFsid(*exported['config'],is_half=False).eval()
    del baseline.enc_q
    baseline.load_state_dict(exported['weight'],strict=True)
    core=shell.generator.state_dict();training=torch.load(OUTPUTS/f'voices/{run}/saves/G_e{epoch}.pth',map_location='cpu',weights_only=True)['model']
    count=0
    for name,value in core.items():
        if name.startswith(('enc_p.cross_attn_layers.','enc_p.text_projection.')): continue
        source=name.replace('enc_p.base_encoder.','enc_p.',1)
        torch.testing.assert_close(value,training[source],rtol=0,atol=0);count+=1
    # Exported V2 weights were rounded to fp16; use the same rounding for waveform parity.
    for name in list(core):
        source=name.replace('enc_p.base_encoder.','enc_p.',1)
        if source in exported['weight']: core[name]=exported['weight'][source].float()
    shell.generator.load_state_dict(core,strict=True)
    phone=torch.randn(1,32,768);lengths=torch.tensor([32]);pitch=torch.full((1,32),110,dtype=torch.long)
    pitchf=torch.full((1,32),220.);sid=torch.tensor([0])
    with torch.no_grad():
        torch.manual_seed(18);expected=baseline.infer(phone,lengths,pitch,pitchf,sid)[0]
        torch.manual_seed(18);actual=shell.generator.infer(phone,lengths,pitch,pitchf,sid,text_strength=0,noise_scale=.66666)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    write_json(folder/'parity.json',dict(core_tensors=count,max_abs_error=float((actual-expected).abs().max()),
        samples=actual.shape[-1],status='Passed same inputs, weights, seed, latent noise; pipeline quality is separate.'))
    print(voice,'passed',flush=True)
