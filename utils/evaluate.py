# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:39:30 2026

@author: Exaud
"""

import torch


@torch.no_grad()
def collect_logits(model, loader, snr, ratio, chan, device, seed=0):
    """Per-round logits (N, K, C) and labels (N,) for one (channel, SNR, ratio) point.
    The seed fixes channel noise and view crops, so every delta in a sweep sees the
    same channel realisations."""
    model.eval()
    torch.manual_seed(seed)
    L, Y = [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        L.append(model.round_logits(x, snr, ratio, chan, view_seed=seed * 100003 + i).cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def early_exit_from_logits(logits, delta):
    """First round tau whose max-softmax >= delta. The last round always exits.
    Returns (predictions (N,), rounds (N,) in 1..K)."""
    N, K, _ = logits.shape
    conf = logits.softmax(-1).max(-1).values
    ok = conf >= delta
    ok[:, -1] = True
    rounds = (ok.long().cumsum(1) == 0).sum(1) + 1
    sel = logits[torch.arange(N), rounds - 1]
    return sel.argmax(-1), rounds


def accuracy(preds, y):
    return (preds == y).float().mean().item()
