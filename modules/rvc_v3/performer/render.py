"""M1 control: generated guide content, source musical pitch, frozen V2 identity."""
import numpy as np
from .contracts import FitError
from .timing import proportional_frames


def guide_time_map(voiced,target_frames):
    """Allocate time to voiced/unvoiced runs. This is NOT learned phone alignment."""
    voiced=np.asarray(voiced,dtype=bool)
    if not len(voiced):raise FitError('Empty pronunciation guide')
    boundaries=np.r_[0,np.flatnonzero(voiced[1:]!=voiced[:-1])+1,len(voiced)]
    runs=[(int(a),int(b),bool(voiced[a])) for a,b in zip(boundaries[:-1],boundaries[1:])]
    lengths=np.array([b-a for a,b,_ in runs])
    lower=[max(1,int(np.ceil(n*(.45 if v else .7)))) for n,(_,_,v) in zip(lengths,runs)]
    upper=[max(lo,int(np.ceil(n*(8. if v else 1.5)))) for lo,n,(_,_,v) in zip(lower,lengths,runs)]
    allocated=proportional_frames(lengths,lower,int(target_frames),upper)
    mapping=np.concatenate([np.linspace(a,b-1,n) for (a,b,_),n in zip(runs,allocated)])
    return mapping,dict(method='bounded_proportional_voicing_v2',guide_frames=len(voiced),target_frames=target_frames,
                        runs=[dict(start=a,end=b,voiced=v,frames=int(n)) for (a,b,v),n in zip(runs,allocated)])


class FrozenRVCGuideRenderer:
    def __init__(self,profile):
        import torch
        from modules.rvc.configs.config import Config
        from modules.rvc.infer.modules.vc.pipeline import VC,load_hubert
        from modules.rvc.infer.lib.rmvpe import RMVPE
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        self.torch=torch;self.config=Config();self.vc=VC(self.config,True)
        self.vc.get_vc(profile['checkpoint']['path'])
        if self.vc.version!='v2' or not self.vc.if_f0:raise ValueError('Performer M1 requires pitch-guided V2')
        for p in self.vc.net_g.parameters():p.requires_grad_(False)
        self.hubert=load_hubert(self.config)
        self.pitch=RMVPE('D:/models/AudioLab/rvc/rmvpe.pt',is_half=False,device=self.config.device)
        self.rate=int(self.vc.tgt_sr);self.hop=int(np.prod(self.vc.cpt['config'][12]))
        if self.rate/self.hop!=100:raise ValueError('M1 requires native 100 Hz V2 checkpoints')

    def render(self,source,source_rate,guide,guide_rate):
        import librosa
        import torch.nn.functional as F
        torch=self.torch
        src=librosa.resample(source,orig_sr=source_rate,target_sr=16000)
        wav=librosa.resample(guide,orig_sr=guide_rate,target_sr=16000)
        guide_f0=self.pitch.infer_from_audio(wav,thred=.03)
        src_f0=self.pitch.infer_from_audio(src,thred=.03)
        tensor=torch.from_numpy(wav).to(self.config.device).unsqueeze(0)
        tensor=tensor.half() if self.config.is_half else tensor.float()
        with torch.no_grad():
            feats=self.hubert.extract_features(source=tensor,padding_mask=torch.zeros_like(tensor,dtype=torch.bool),output_layer=12)[0]
            feats=F.interpolate(feats.transpose(1,2),size=len(guide_f0),mode='linear',align_corners=False).transpose(1,2)[0].float().cpu().numpy()
        count=max(1,round(len(source)/source_rate*100))
        mapping,fit=guide_time_map(guide_f0>0,count)
        features=np.stack([np.interp(mapping,np.arange(len(feats)),feats[:,i]) for i in range(768)],axis=1).astype('float32')
        voiced=(guide_f0[np.rint(mapping).astype(int)]>0)
        good=np.flatnonzero(src_f0>0)
        if not len(good) and voiced.any():raise FitError('No reliable source pitch; cannot preserve musical contour')
        pitch=np.interp(np.linspace(0,len(src_f0)-1,count),good,src_f0[good]) if len(good) else np.zeros(count)
        pitch=np.where(voiced,pitch,0).astype('float32')
        mel=1127*np.log1p(pitch/700);low=1127*np.log1p(50/700);high=1127*np.log1p(1100/700)
        coarse=np.where(pitch>0,(mel-low)*254/(high-low)+1,1).round().clip(1,255).astype('int64')
        dtype=next(self.vc.net_g.parameters()).dtype;device=self.config.device
        with torch.no_grad():
            result=self.vc.net_g.infer(torch.from_numpy(features)[None].to(device=device,dtype=dtype),
                torch.tensor([count],device=device),torch.from_numpy(coarse)[None].to(device),
                torch.from_numpy(pitch)[None].to(device),torch.tensor([0],device=device))[0][0,0].float().cpu().numpy()
        desired=round(len(source)/source_rate*self.rate)
        # Only sample-grid rounding is allowed; never trim a generated phrase to hide overflow.
        if abs(len(result)-desired)>self.hop:raise FitError('Renderer timeline exceeds a single frame rounding error')
        if len(result)>desired:result=result[:desired]
        else:result=np.pad(result,(0,desired-len(result)))
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite rendered phrase')
        fit.update(frame_rate=100,renderer_sample_rate=self.rate,renderer_hop=self.hop,
                   guide_voiced_fraction=float(np.mean(guide_f0>0)),source_voiced_fraction=float(np.mean(src_f0>0)),
                   rendered_voiced_fraction=float(np.mean(pitch>0)))
        return result,self.rate,fit
