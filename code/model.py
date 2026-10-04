"""Backbone creation, freezing, optimizer groups and complexity."""
from __future__ import annotations
import copy

SUGGESTED_BACKBONES = {"resnet50":"resnet50", "resnext50":"resnext50_32x4d", "convnext_tiny":"convnext_tiny", "deit_small":"deit_small_patch16_224", "swin_tiny":"swin_tiny_patch4_window7_224", "efficientnet_b0":"efficientnet_b0", "mobilenetv3":"mobilenetv3_large_100"}

def _classifier_params(model):
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}

def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init="finetune"):
    import timm
    if init not in ("scratch", "frozen", "finetune"): raise ValueError(f"unknown init: {init}")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch", num_classes=num_classes, drop_rate=drop_rate)
    model.pretrained_tag = copy.deepcopy(getattr(model, "pretrained_cfg", {}))
    if init == "frozen": freeze_backbone(model)
    return model

def freeze_backbone(model):
    import torch.nn as nn
    head_ids = _classifier_params(model)
    for p in model.parameters(): p.requires_grad = id(p) in head_ids
    model._backbone_frozen = True
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()

def keep_frozen_bn_eval(model):
    import torch.nn as nn
    if getattr(model, "_backbone_frozen", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = _classifier_params(model); backbone_decay=[]; backbone_no_decay=[]; head=[]
    for p in model.parameters():
        if not p.requires_grad: continue
        if id(p) in head_ids: head.append(p)
        elif p.ndim <= 1: backbone_no_decay.append(p)
        else: backbone_decay.append(p)
    groups=[]
    if backbone_decay: groups.append({"params":backbone_decay,"lr":lr_backbone,"weight_decay":weight_decay})
    if backbone_no_decay: groups.append({"params":backbone_no_decay,"lr":lr_backbone,"weight_decay":0.})
    if head: groups.append({"params":head,"lr":lr_head,"weight_decay":weight_decay})
    return groups

def count_params(model): return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size=224):
    import torch
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e: raise RuntimeError("install fvcore to count GMACs") from e
    device = next(model.parameters()).device; dummy = torch.zeros(1,3,img_size,img_size,device=device)
    return FlopCountAnalysis(model, dummy).total() / 1e9
