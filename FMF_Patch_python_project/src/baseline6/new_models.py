"""Six original forecasting backbones with only two ordered quantile heads."""
import sys, importlib, math
from pathlib import Path
from types import SimpleNamespace, MethodType
from portable import torch, np
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
KINDS=['TimesNet','Informer','iTransformer','Autoformer','PatchTST','DLinear']
CONFIG=dict(d_model=128,e_layers=2,d_layers=1,d_ff=128,n_heads=8,factor=1,dropout=.1,
    embed='timeF',freq='h',activation='gelu',output_attention=False,channel_independence=1,
    deformable_grouping=True,use_norm=1,classify=False,num_class=2,top_k=5,num_kernels=6,
    moving_avg=25,individual=False,distil=True,seg_len=6,task_name='long_term_forecast',
    feature_type='MS',enc_in=49,dec_in=49,c_out=49,augmentations=3,decomp_method='moving_avg',
    down_sampling_window=2,down_sampling_layers=2,down_sampling_method='avg',seq_len=72,label_len=36,pred_len=3)
def seed_all(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
class WindowLocalTimesBlock(nn.Module):
    def __init__(self,original):
        super().__init__();self.conv=original.conv;self.k=original.k
    def forward(self,x):
        n,steps,width=x.shape;scores=torch.fft.rfft(x,dim=1).abs().mean(-1)
        rank=scores.clone();rank[:,0]=-torch.inf;ids=rank.topk(self.k,dim=-1).indices
        periods=torch.div(steps,ids,rounding_mode='floor');weights=scores.gather(1,ids).softmax(-1)
        result=torch.zeros_like(x)
        for period in torch.unique(periods).tolist():
            selected=periods.eq(period);rows=selected.any(-1).nonzero(as_tuple=False).flatten()
            z=x.index_select(0,rows);length=((steps+period-1)//period)*period
            if length>steps:z=F.pad(z,(0,0,0,length-steps))
            z=z.reshape(len(rows),length//period,period,width).permute(0,3,1,2).contiguous()
            z=self.conv(z).permute(0,2,3,1).reshape(len(rows),length,width)[:,:steps]
            coefficient=(weights*selected).sum(-1).index_select(0,rows)
            result=result.index_add(0,rows,z*coefficient[:,None,None])
        return x+result
def deterministic_prob_qk(self,Q,K,sample_k,n_top):
    B,H,LK,E=K.shape;LQ=Q.shape[2]
    # Uniform stratified circular key sample, identical for every window and replay.
    index=(torch.arange(LQ,device=Q.device)[:,None]+torch.div(torch.arange(sample_k,device=Q.device)*LK,sample_k,rounding_mode='floor')[None,:])%LK
    sample=K[:,:,index,:];scores=(Q.unsqueeze(-2)*sample).sum(-1)
    sparsity=scores.max(-1).values-scores.sum(-1)/LK;top=sparsity.topk(n_top,sorted=False).indices
    reduced=Q.gather(2,top[...,None].expand(-1,-1,-1,E))
    return reduced@K.transpose(-2,-1),top
class Baseline(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.q50_head=nn.Linear(49,1);self.delta90_head=nn.Linear(49,1)
        with torch.no_grad():self.q50_head.weight.zero_();self.q50_head.weight[0,-1]=1.;self.q50_head.bias.zero_()
    def forward(self,x,xmark,ymark):
        decoder=x.new_zeros((len(x),39,49));decoder[:,:36]=x[:,-36:]
        hidden=self.backbone(x,xmark,decoder,ymark);q=self.q50_head(hidden).squeeze(-1)
        return dict(q50=q,q90=q+F.softplus(self.delta90_head(hidden).squeeze(-1)))
def build(kind,seed=20260908):
    assert kind in KINDS;seed_all(seed)
    mod=importlib.import_module('models.'+kind);cfg=SimpleNamespace(**CONFIG)
    core=mod.Model(cfg,patch_len=12,stride=6) if kind=='PatchTST' else mod.Model(cfg)
    if kind=='TimesNet':core.model=nn.ModuleList([WindowLocalTimesBlock(x) for x in core.model])
    if kind=='Informer':
        from layers.SelfAttention_Family import ProbAttention
        for layer in core.modules():
            if isinstance(layer,ProbAttention):layer._prob_QK=MethodType(deterministic_prob_qk,layer)
    net=Baseline(core)
    assert not any('80' in k for k in net.state_dict())
    return net
