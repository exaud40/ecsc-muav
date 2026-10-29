# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 10:24:50 2026

@author: Exaud
"""

import os

import torch
import torch.optim as optim
from tqdm import tqdm

from config import run_name
from data.dataloader import get_loaders
from models.net import MultiUAV
from utils.common import get_device, save_json, set_seed
from utils.evaluate import accuracy, collect_logits


def validate(model, loader, cfg, device):
    """Validation accuracy with all K UAVs at the reference point."""
    logits, y = collect_logits(model, loader, cfg.ref_snr, 1.0, cfg.ref_chan, device, seed=0)
    return accuracy(logits[:, -1].argmax(-1), y)


def train(cfg):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    loaders, num_classes = get_loaders(cfg)
    model = MultiUAV(cfg, num_classes).to(device)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad],
                     lr=cfg.lr, weight_decay=cfg.weight_decay)

    name = run_name(cfg)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(cfg.ckpt_dir, f"{name}.pt")
    best, history = -1.0, []

    for epoch in range(cfg.epochs):
        model.train()
        tot, n, correct, m = 0.0, 0, 0, 0
        for x, y in tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x, y)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += out["loss"].item() * y.size(0)
            n += y.size(0)
            correct += (out["logits"].argmax(1) == y).sum().item()
            m += y.size(0)
        val_acc = validate(model, loaders["val"], cfg, device)
        history.append({"epoch": epoch, "train_loss": tot / n, "train_acc": correct / m, "val_acc": val_acc})
        print(f"[{name}] epoch {epoch + 1}: loss {tot / n:.4f} | train acc {correct / m:.4f} | val acc {val_acc:.4f}")
        if val_acc > best:                      # checkpoint selected on VALIDATION accuracy only
            best = val_acc
            torch.save({"model": model.trainable_state_dict(), "val_acc": best,
                        "epoch": epoch, "num_classes": num_classes}, ckpt)

    save_json(history, os.path.join(cfg.results_dir, "logs", f"{name}_log.json"))
    print(f"[{name}] best val acc {best:.4f} -> {ckpt}")
    return ckpt
