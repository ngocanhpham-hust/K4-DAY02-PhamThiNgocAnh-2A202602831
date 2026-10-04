"""Single training entry point for all DeepWeeds experiments."""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time, types
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

@dataclass
class Config:
    exp_id:str="T00"; seed:int=0; fold:int=0; backbone:str="resnet50"; init:str="finetune"; drop_rate:float=0.
    img_size:int=224; aug:str="basic"; sampler:str|None=None; mix:str|None=None; mix_alpha:float=1.
    loss:str="ce"; label_smoothing:float=0.; focal_gamma:float=2.; class_weight_beta:float|None=None
    epochs:int=12; batch_size:int=64; lr_backbone:float=1e-4; lr_head:float=1e-3; weight_decay:float=.05
    warmup_epochs:float=1.; ema_decay:float|None=None; amp:bool=True; num_workers:int=2
    images_dir:str="data/images"; labels_dir:str="data/labels"; out_dir:str="runs"; pred_dir:str="predictions"
    save_test_predictions:bool=False

def run_dir(cfg): return Path(cfg.out_dir)/cfg.exp_id/f"seed{cfg.seed}"
def pred_path(cfg,split): return Path(cfg.pred_dir)/f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"

def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def build_optimizer(model,cfg):
    import torch
    from model import param_groups
    return torch.optim.AdamW(param_groups(model,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay))

def build_scheduler(optimizer,cfg,steps_per_epoch):
    import torch
    total=max(1,cfg.epochs*steps_per_epoch); warm=int(cfg.warmup_epochs*steps_per_epoch)
    def factor(step):
        if warm and step < warm: return max(1e-8,(step+1)/warm)
        progress=(step-warm)/max(1,total-warm); return .5*(1+math.cos(math.pi*min(progress,1.)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)

class EMA:
    def __init__(self,model,decay): self.decay=decay; self.module=copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in self.module.parameters()]
    def update(self,model):
        import torch
        with torch.no_grad():
            source=model.state_dict()
            for key,value in self.module.state_dict().items():
                src=source[key].detach()
                if value.is_floating_point(): value.mul_(self.decay).add_(src,alpha=1-self.decay)
                else: value.copy_(src)

def train_one_epoch(model,loader,criterion,optimizer,scheduler,scaler,cfg,device,ema=None):
    import torch
    from losses import mix_batch,mixed_loss
    from model import keep_frozen_bn_eval
    model.train(); keep_frozen_bn_eval(model); total=0.; seen=0; start=time.perf_counter()
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); targets=y
        if cfg.mix: x,targets=mix_batch(x,y,cfg.mix_alpha,cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=cfg.amp and device.type=="cuda"):
            logits=model(x); loss=mixed_loss(criterion,logits,targets) if cfg.mix else criterion(logits,y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema: ema.update(model)
        total += float(loss.detach())*x.size(0); seen += x.size(0)
    return {"train_loss":total/max(seen,1),"lr":optimizer.param_groups[0]["lr"],"epoch_seconds":time.perf_counter()-start}

def evaluate(model,loader,criterion,device):
    import torch
    model.eval(); names=[]; ys=[]; logits_all=[]; total=0.; seen=0
    with torch.inference_mode():
        for x,y,batch_names in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); logits=model(x); loss=criterion(logits,y)
            total += float(loss)*x.size(0); seen += x.size(0); names.extend(batch_names); ys.append(y.cpu()); logits_all.append(logits.float().cpu())
    return names,torch.cat(ys).numpy(),torch.cat(logits_all).numpy(),total/max(seen,1)

def plot_curves(history,path,title):
    import matplotlib.pyplot as plt
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); epochs=[r["epoch"] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot(epochs,[r["train_loss"] for r in history],label="train"); axes[0].plot(epochs,[r["val_loss"] for r in history],label="val"); axes[0].set(xlabel="Epoch",ylabel="Loss",title="Loss"); axes[0].legend()
    axes[1].plot(epochs,[r["val_macro_f1"] for r in history],label="macro-F1"); axes[1].set(xlabel="Epoch",ylabel="Score",title="Validation macro-F1"); axes[1].legend(); fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def _probs(logits):
    z=logits-logits.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def run(cfg):
    import pandas as pd, torch
    from dataset import build_transforms,check_split,load_split,make_loader
    from eval import compute_metrics,save_predictions
    from losses import build_criterion,class_weights
    from model import build_model,count_gmacs,count_params
    set_seed(cfg.seed); out=run_dir(cfg); out.mkdir(parents=True,exist_ok=True); (out/"config.json").write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    train_df,val_df,test_df=load_split(cfg.labels_dir,cfg.fold); split_info=check_split(train_df,val_df,test_df,cfg.images_dir); (out/"split.json").write_text(json.dumps(split_info,indent=2),encoding="utf-8")
    train_loader=make_loader(train_df,cfg.images_dir,build_transforms(True,cfg.img_size,cfg.aug),cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    val_loader=make_loader(val_df,cfg.images_dir,build_transforms(False,cfg.img_size),cfg.batch_size,False,None,cfg.num_workers)
    test_loader=make_loader(test_df,cfg.images_dir,build_transforms(False,cfg.img_size),cfg.batch_size,False,None,cfg.num_workers) if cfg.save_test_predictions else None
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_model(cfg.backbone,True,9,cfg.drop_rate,cfg.init).to(device)
    weight=None
    if cfg.loss=="ce_weighted" or cfg.class_weight_beta is not None:
        counts=train_df["Label"].value_counts().reindex(range(9)).to_numpy(); weight=class_weights(counts,0. if cfg.class_weight_beta is None else cfg.class_weight_beta).to(device)
    criterion=build_criterion(cfg.loss,smoothing=cfg.label_smoothing,gamma=cfg.focal_gamma,weight=weight)
    optimizer=build_optimizer(model,cfg); scheduler=build_scheduler(optimizer,cfg,len(train_loader)); scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda"); ema=EMA(model,cfg.ema_decay) if cfg.ema_decay else None
    history=[]; best=-1.; best_epoch=0; checkpoint=out/"best.pt"
    for epoch in range(1,cfg.epochs+1):
        row=train_one_epoch(model,train_loader,criterion,optimizer,scheduler,scaler,cfg,device,ema); eval_model=ema.module if ema else model
        _,y,logits,val_loss=evaluate(eval_model,val_loader,criterion,device); probs=_probs(logits); metrics=compute_metrics(y,probs.argmax(1),probs); row.update({"epoch":epoch,"val_loss":val_loss,"val_macro_f1":metrics["macro_f1"],"val_top1":metrics["top1"]}); history.append(row); pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        if metrics["macro_f1"] > best: best=metrics["macro_f1"]; best_epoch=epoch; torch.save(eval_model.state_dict(),checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); names,y,logits,_=evaluate(model,val_loader,criterion,device); np.save(out/"val_logits.npy",logits); probs=_probs(logits); save_predictions(pred_path(cfg,"val"),names,y,probs)
    val_metrics=compute_metrics(y,probs.argmax(1),probs)
    if test_loader is not None:
        names_t,y_t,logits_t,_=evaluate(model,test_loader,criterion,device); np.save(out/"test_logits.npy",logits_t); save_predictions(pred_path(cfg,"test"),names_t,y_t,_probs(logits_t))
    plot_curves(history,Path("curves")/f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png",f"{cfg.exp_id} · {cfg.backbone}")
    try: gmac=count_gmacs(model,cfg.img_size)
    except (RuntimeError,ImportError): gmac=float("nan")
    summary={"best_epoch":best_epoch,"val_macro_f1":val_metrics["macro_f1"],"val_top1":val_metrics["top1"],"params_m":count_params(model),"gmac":gmac,"mean_epoch_seconds":float(np.mean([r["epoch_seconds"] for r in history]))}; (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary

def parse_overrides(pairs):
    fmap={f.name:f for f in fields(Config)}; result={}
    for pair in pairs:
        if "=" not in pair: raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        key,value=pair.split("=",1)
        if key not in fmap: raise ValueError(f"unknown Config field: {key}")
        default=getattr(Config(),key); low=value.lower()
        if low in ("none","null"): parsed=None
        elif isinstance(default,bool):
            if low not in ("true","false","1","0","yes","no"): raise ValueError(f"invalid bool for {key}: {value}")
            parsed=low in ("true","1","yes")
        elif isinstance(default,int): parsed=int(value)
        elif isinstance(default,float): parsed=float(value)
        else:
            annotation=str(fmap[key].type)
            if default is None and ("float" in annotation): parsed=float(value)
            else: parsed=value
        result[key]=parsed
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--set",nargs="*",default=[]); args=parser.parse_args(); print(json.dumps(run(Config(**parse_overrides(args.set))),indent=2))

if __name__=="__main__": main()
