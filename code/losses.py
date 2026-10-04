"""Losses and batch-level Mixup/CutMix."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def build_criterion(kind="ce", **kw):
    weight = kw.get("weight")
    if kind == "ce": return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls": return LabelSmoothingCE(kw.get("smoothing", .1), weight)
    if kind == "focal": return FocalLoss(kw.get("gamma", 2.), kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None: raise ValueError("ce_weighted requires weight")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"unknown loss: {kind}")

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=.1, weight=None):
        super().__init__()
        if not 0 <= smoothing < 1: raise ValueError("smoothing must be in [0,1)")
        self.smoothing = smoothing; self.register_buffer("weight", weight)
    def forward(self, logits, target): return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super().__init__()
        if gamma < 0: raise ValueError("gamma must be non-negative")
        self.gamma = gamma; self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None)
    def forward(self, logits, target):
        logpt = F.log_softmax(logits, 1).gather(1, target[:, None]).squeeze(1); pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None: loss = loss * self.alpha[target]
        return loss.mean()

def class_weights(counts, beta=0.):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or torch.any(counts <= 0): raise ValueError("counts must be positive 1-D")
    if beta == 0: weights = counts.reciprocal()
    elif 0 < beta < 1: weights = (1 - beta) / (1 - torch.pow(beta, counts))
    else: raise ValueError("beta must be in [0,1)")
    return (weights / weights.mean()).float()

def mix_batch(x, y, alpha=1., mode="cutmix"):
    if alpha <= 0: raise ValueError("alpha must be positive")
    lam = float(np.random.beta(alpha, alpha)); perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup": mixed = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        _, _, h, w = x.shape; ratio = np.sqrt(1 - lam); cut_w, cut_h = int(w * ratio), int(h * ratio)
        cx, cy = np.random.randint(w), np.random.randint(h); x1, x2 = max(cx-cut_w//2, 0), min(cx+cut_w//2, w); y1, y2 = max(cy-cut_h//2, 0), min(cy+cut_h//2, h)
        mixed = x.clone(); mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]; lam = 1 - ((x2-x1)*(y2-y1)/(w*h))
    else: raise ValueError("mode must be mixup or cutmix")
    return mixed, (y, y[perm], float(lam))

def mixed_loss(criterion, logits, targets):
    ya, yb, lam = targets; return lam * criterion(logits, ya) + (1-lam) * criterion(logits, yb)
