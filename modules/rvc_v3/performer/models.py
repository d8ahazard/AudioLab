"""Trainable hypotheses. Random weights are never accepted for inference."""
import torch
from torch import nn


class DeliveryPlanner(nn.Module):
    def __init__(self,vocab_size,performers,width=256,layers=6,heads=8):
        super().__init__()
        self.tokens=nn.Embedding(vocab_size,width,padding_idx=0)
        self.performers=nn.Embedding(performers,width)
        self.context=nn.Linear(8,width)
        self.position=nn.Embedding(2048,width)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,1024,batch_first=True,norm_first=True),layers)
        self.rap=nn.Linear(width,7);self.singing=nn.Linear(width,7)

    def forward(self,tokens,performer,structure,mode,padding_mask=None):
        if tokens.shape[1]>2048:raise ValueError('Split into phrase windows before planning')
        if tokens.shape[1]==0 or (padding_mask is not None and padding_mask.all(1).any()):raise ValueError('Every phrase needs nonpadding tokens')
        x=self.tokens(tokens)+self.performers(performer)[:,None]+self.context(structure)
        x=x+self.position(torch.arange(tokens.shape[1],device=tokens.device))[None]
        x=self.encoder(x,src_key_padding_mask=padding_mask)
        rap=self.rap(x);singing=self.singing(x)
        result=torch.where(mode[:,None,None].bool(),singing,rap)
        # duration logit, stress, breath, pitch gesture, energy, articulation, texture
        return result.masked_fill(padding_mask[:,:,None],0) if padding_mask is not None else result


class AcousticBridge(nn.Module):
    def __init__(self,width=256,layers=6,heads=8):
        super().__init__()
        self.guide=nn.Linear(768,width);self.condition=nn.Linear(8,width)
        self.memory=nn.Linear(768,width)
        self.position=nn.Embedding(4096,width)
        self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,1024,batch_first=True,norm_first=True),layers)
        self.out=nn.Linear(width,768)

    def forward(self,guide,expression,target_memory,padding_mask=None):
        if guide.shape[1]>4096:raise ValueError('Bridge requires bounded phrase batches')
        if guide.shape[1]==0 or (padding_mask is not None and padding_mask.all(1).any()):raise ValueError('Every phrase needs nonpadding frames')
        x=self.guide(guide)+self.condition(expression)+self.memory(target_memory)
        x=x+self.position(torch.arange(guide.shape[1],device=guide.device))[None]
        x=self.blocks(x,src_key_padding_mask=padding_mask)
        out=self.out(x)
        return out.masked_fill(padding_mask[:,:,None],0) if padding_mask is not None else out
