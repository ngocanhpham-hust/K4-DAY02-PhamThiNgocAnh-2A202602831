"""Reproducible forward latency measurement."""
from __future__ import annotations
import time
import numpy as np

def bench(fn,warmup=10,iters=100,sync=None):
    if warmup < 10 or iters < 50: raise ValueError("warmup >= 10 and iters >= 50 required")
    for _ in range(warmup): fn()
    values=[]
    for _ in range(iters):
        if sync: sync()
        start=time.perf_counter(); fn()
        if sync: sync()
        values.append((time.perf_counter()-start)*1000)
    return {"p50":float(np.percentile(values,50)),"p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"mean":float(np.mean(values)),"n":iters}

def latency_report(model,batch_size,img_size,dtype="fp32",device="cuda",warmup=10,iters=100):
    import torch
    if dtype not in ("fp32","amp","fp16"): raise ValueError("dtype must be fp32, amp, or fp16")
    dev=torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    model=model.to(dev).eval(); x=torch.randn(batch_size,3,img_size,img_size,device=dev)
    if dtype == "fp16": model=model.half(); x=x.half()
    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type,enabled=dtype=="amp"): model(x)
    stats=bench(fn,warmup,iters,torch.cuda.synchronize if dev.type=="cuda" else None)
    stats.update({"gpu":torch.cuda.get_device_name(dev) if dev.type=="cuda" else "CPU","dtype":dtype,"batch":batch_size,"img_size":img_size,"images_per_s":batch_size/(stats["p50"]/1000),"torch":torch.__version__,"preprocessing":False})
    return stats

def tta_latency(model,k_views,**kw):
    import torch
    if k_views < 1: raise ValueError("k_views must be positive")
    device=kw.get("device","cuda"); batch_size=kw["batch_size"]; img_size=kw["img_size"]; dtype=kw.get("dtype","fp32")
    dev=torch.device(device); model=model.to(dev).eval(); x=torch.randn(batch_size,3,img_size,img_size,device=dev)
    if dtype=="fp16": model=model.half(); x=x.half()
    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type,enabled=dtype=="amp"):
            for i in range(k_views): model(torch.flip(x,(-1,)) if i%2 else x)
    stats=bench(fn,kw.get("warmup",10),kw.get("iters",100),torch.cuda.synchronize if dev.type=="cuda" else None); stats["k_views"]=k_views; return stats
