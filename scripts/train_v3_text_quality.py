"""Small frozen-backbone lyric-adapter feasibility experiment, independent of V2 tuning."""
import json
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rvc_quality import ROOT, write_json, fingerprint

def main():
    import numpy as np
    import torch
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel
    from modules.rvc_v3.io.checkpoint_io import load_inference_model,save_inference_model
    from modules.rvc_v3.configs.v3_config import RVCV3Config
    from modules.rvc_v3.inference.pipeline import RVCV3Pipeline
    from modules.rvc_v3.models.content_encoders import HuBERTEncoder
    from modules.rvc.infer.lib.rmvpe import RMVPE
    from modules.rvc.infer.lib.train.mel_processing import spectrogram_torch
    from modules.rvc.infer.lib.train.losses import kl_loss
    from modules.rvc_v3.text_tokens import encode_char_v1
    torch.set_num_threads(4);random.seed(20260924);np.random.seed(20260924);torch.manual_seed(20260924)
    folder=ROOT/'v3_text/huxlxy';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'seed_training/huxlxy/manifest.json').read_text())
    groups={}
    for row in manifest: groups.setdefault(row['source'],[]).append(row)
    selected=[]
    for source,rows in sorted(groups.items()):
        ranked=sorted(rows,key=lambda r:float(np.mean(sf.read(r['clip']['path'],dtype='float32')[0]**2)),reverse=True)
        selected.extend(ranked[:2])
    write_json(folder/'sources.json',selected)
    asr=WhisperModel('distil-large-v3',device='cuda',compute_type='float16',
                    download_root='D:/models/hf-cache/hub',local_files_only=True)
    for row in selected:
        path=Path(row['clip']['path']);dest=folder/'lyrics'/f'{path.stem}.json'
        if dest.exists(): continue
        segments,_=asr.transcribe(str(path),language='en',word_timestamps=True,condition_on_previous_text=False)
        write_json(dest,dict(source=fingerprint(path),reviewed=False,
            words=[dict(start=w.start,end=w.end,text=w.word) for seg in segments for w in (seg.words or [])]))
    del asr;torch.cuda.empty_cache()
    hubert=HuBERTEncoder('D:/models/AudioLab/rvc/hubert_base.pt',device='cuda')
    pitch_model=RMVPE('D:/models/AudioLab/rvc/rmvpe.pt',is_half=False,device='cuda')
    for row in selected:
        path=Path(row['clip']['path']);dest=folder/'features'/f'{path.stem}.pt'
        if dest.exists(): continue
        x,_=librosa.load(path,sr=16000)
        with torch.no_grad(): content=hubert.extract_features(torch.from_numpy(x)[None]).squeeze(0).cpu()
        f0=pitch_model.infer_from_audio(x,thred=.03)
        wave,_=librosa.load(path,sr=48000)
        n=len(wave)//480
        content=torch.nn.functional.interpolate(content.T[None],size=n,mode='nearest')[0].T
        f0=np.interp(np.arange(n)*.01,np.arange(len(f0))*.01,f0).astype('float32')
        mel=1127*np.log1p(f0/700);lo=1127*np.log1p(50/700);hi=1127*np.log1p(1100/700)
        coarse=np.rint(np.clip((mel-lo)*254/(hi-lo)+1,1,255)).astype('int64')
        dest.parent.mkdir(exist_ok=True)
        torch.save(dict(content=content,pitch=torch.from_numpy(coarse),pitchf=torch.from_numpy(f0),wave=torch.from_numpy(wave)),dest)
    del hubert,pitch_model;torch.cuda.empty_cache()
    base=ROOT/'v3/huxlxy/f0G48k.safetensors'
    gen,txt,metadata=load_inference_model(str(base),device='cpu')
    config=RVCV3Config.from_dict(metadata['config']);config.text_tokenizer='char_v1'
    shell=RVCV3Pipeline.__new__(RVCV3Pipeline);shell.config=config;shell.device='cuda'
    shell._init_generator();shell._init_text_encoder()
    model=shell.generator;text=shell.text_encoder
    model.load_state_dict(gen,strict=True);text.load_state_dict(txt,strict=True)
    core={k:v.clone() for k,v in gen.items() if not k.startswith(('enc_p.cross_attn_layers.','enc_p.text_projection.'))}
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith(('enc_p.cross_attn_layers.','enc_p.text_projection.')))
    model.eval();text.train()
    params=[p for p in model.parameters() if p.requires_grad]+list(text.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-5)
    data=[]
    for row in selected:
        stem=Path(row['clip']['path']).stem
        features=torch.load(folder/'features'/f'{stem}.pt',weights_only=True)
        words=json.loads((folder/'lyrics'/f'{stem}.json').read_text())['words']
        if words:data.append((features,words))
    if not data: raise RuntimeError('No usable text-labelled training clips')
    history=[]
    for step in range(1,201):
        feat,words=random.choice(data);frames=96
        anchors=[w for w in words if w['text'].strip()]
        anchor=random.choice(anchors)
        start=max(0,min(int(anchor['start']*100),len(feat['content'])-frames));end=start+frames
        lyrics=' '.join(w['text'] for w in words if w['start']<end*.01 and w['end']>start*.01)
        ids=encode_char_v1(lyrics)
        if not ids: raise RuntimeError('Empty transcript crop')
        phone=feat['content'][start:end][None].cuda();pitch=feat['pitch'][start:end][None].cuda()
        wave=feat['wave'][start*480:end*480][None].cuda()
        spec=spectrogram_torch(wave,2048,48000,480,2048,center=False)
        n=min(phone.shape[1],spec.shape[-1]);phone=phone[:,:n];pitch=pitch[:,:n];spec=spec[:,:,:n]
        lengths=torch.tensor([n],device='cuda');sid=torch.tensor([0],device='cuda')
        with torch.no_grad():
            g=model.emb_g(sid).unsqueeze(-1)
            z,mq,logq,mask=model.enc_q(spec,lengths,g=g)
            zprior=model.flow(z,mask,g=g)
            original,_,_=model.enc_p(phone,pitch,lengths)
        tokens=torch.tensor([ids],device='cuda');textmask=torch.zeros_like(tokens,dtype=torch.bool)
        features=text(tokens,textmask)
        mean,logs,mask=model.enc_p(phone,pitch,lengths,text_features=features,text_mask=textmask,text_strength=.35)
        loss=kl_loss(zprior,logq,mean,logs,mask)+.05*((mean-original)**2).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite adapter loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        history.append(dict(step=step,loss=float(loss)))
        if step%25==0: print('Adapter step',step,'loss',float(loss),flush=True)
        if step in (50,100,200): save_inference_model(folder/f'adapter_{step}',model,text,config.to_dict(),iteration=step)
    for name,value in core.items():torch.testing.assert_close(model.state_dict()[name].cpu(),value,rtol=0,atol=0)
    write_json(folder/'training.json',dict(steps=200,seed=20260924,frozen_core_tensors=len(core),history=history,
        objective='Frozen V2 backbone, text prior adapter KL + residual regularization; no vocoder or speaker-weight updates.',
        limitation='Small feasibility pilot using unreviewed ASR labels; not a validated production model.'))

if __name__=='__main__':main()
