"""Inference-time augmentation, calibration, ensembles and BN fusion."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def predict_logits(model, loader, device, view=None):
    model.eval(); names=[]; ys=[]; outs=[]
    with torch.inference_mode():
        for x,y,batch_names in loader:
            x=x.to(device, non_blocking=True); x=view(x) if view else x
            outs.append(model(x).float().cpu()); ys.append(y.cpu()); names.extend(batch_names)
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy()

def view_identity(x): return x
def view_hflip(x): return torch.flip(x, (-1,))

def views_multicrop(x, crop):
    h,w=x.shape[-2:]
    if crop > min(h,w): raise ValueError("crop exceeds input size")
    positions=((0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2))
    return [x[..., top:top+crop,left:left+crop] for top,left in positions]

def views_multiscale(x, sizes): return [F.interpolate(x,size=(s,s),mode="bilinear",align_corners=False,antialias=True) for s in sizes]

def _softmax_np(z):
    z=np.asarray(z,dtype=np.float64); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def aggregate_views(logits_per_view, space="prob"):
    arrays=[np.asarray(z) for z in logits_per_view]
    if not arrays or any(z.shape != arrays[0].shape for z in arrays): raise ValueError("views must have identical non-empty shapes")
    if space == "prob": p=np.mean([_softmax_np(z) for z in arrays],axis=0); return p/p.sum(1,keepdims=True)
    if space == "logit": return _softmax_np(np.mean(arrays,axis=0))
    raise ValueError("space must be prob or logit")

def ensemble_probs(list_of_probs):
    arr=[np.asarray(p,dtype=np.float64) for p in list_of_probs]
    if not arr or any(p.shape != arr[0].shape for p in arr): raise ValueError("probabilities must have identical shapes")
    out=np.mean(arr,axis=0); return out/out.sum(1,keepdims=True)

def fit_temperature(val_logits, val_labels):
    logits=torch.as_tensor(val_logits,dtype=torch.float64); labels=torch.as_tensor(val_labels,dtype=torch.long); log_t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=100,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=F.cross_entropy(logits/log_t.exp().clamp(1e-3,1e3),labels); loss.backward(); return loss
    opt.step(closure); return float(log_t.detach().exp().clamp(1e-3,1e3))

def apply_temperature(logits,T):
    if T <= 0: raise ValueError("T must be positive")
    return _softmax_np(np.asarray(logits)/T)

def _fuse_sequence(module):
    children=list(module.named_children())
    for name,child in children: _fuse_sequence(child)
    children=list(module.named_children())
    for i in range(len(children)-1):
        n1,m1=children[i]; n2,m2=children[i+1]
        if isinstance(m1,nn.Conv2d) and isinstance(m2,nn.BatchNorm2d):
            fused=torch.nn.utils.fusion.fuse_conv_bn_eval(m1,m2); setattr(module,n1,fused); setattr(module,n2,nn.Identity())

def fuse_conv_bn(model):
    fused=copy.deepcopy(model).eval(); _fuse_sequence(fused); return fused
