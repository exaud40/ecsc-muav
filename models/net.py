# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:52:06 2026

@author: Exaud
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel import Channel
from models.decoder import FusionDecoder
from models.encoder import UAVEncoder
from models.energy import EnergyNet, contrastive_energy_loss
from models.feature_extractor import CLIPTeacher
from models.views import make_views


class MultiUAV(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg, self.K = cfg, cfg.K
        self.use_teacher = cfg.variant in ("ecsc", "cosine_distill")
        self.encoders = nn.ModuleList([UAVEncoder(cfg.bottleneck_dim) for _ in range(cfg.K)])
        self.channel = Channel(cfg.rician_k_lin)
        self.decoder = FusionDecoder(cfg.K, cfg.bottleneck_dim, num_classes,
                                     cfg.teacher_dim if self.use_teacher else None)
        self.clip = CLIPTeacher(cfg.clip_name, cfg.teacher_dim) if self.use_teacher else None
        self.energy_net = EnergyNet(cfg.teacher_dim) if cfg.variant == "ecsc" else None

    # keep the frozen teacher in eval mode
    def train(self, mode=True):
        super().train(mode)
        if self.clip is not None:
            self.clip.eval()
        return self

    # checkpoints exclude the (large, frozen, downloadable) teacher
    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("clip.")}

    def load_trainable_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("clip.") for k in missing), (missing, unexpected)

    # ---- building blocks -------------------------------------------------
    def _views(self, x, view_seed):
        gen = None
        if view_seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(int(view_seed))
        return make_views(x, self.K, self.cfg.view_mode, generator=gen)

    def _transmit(self, views, snr, ratio, chan):
        ws, ys = [], []
        for enc, v in zip(self.encoders, views):
            w = Channel.power_normalize(enc(v))
            ws.append(w)
            ys.append(self.channel(w, snr, ratio, chan))
        return ws, ys

    def _decode(self, ys, keep):
        feats = [y if k else torch.zeros_like(y) for y, k in zip(ys, keep)]
        return self.decoder(torch.cat(feats, dim=1))

    # ---- inference: logits after every round -----------------------------
    def round_logits(self, x, snr, ratio=1.0, chan="awgn", view_seed=None):
        """(B, K, C): logits using the first tau UAVs, tau = 1..K. One channel
        realisation is shared by all rounds, so early exit is evaluated fairly."""
        _, ys = self._transmit(self._views(x, view_seed), snr, ratio, chan)
        outs = [self._decode(ys, [k < tau for k in range(self.K)])[0] for tau in range(1, self.K + 1)]
        return torch.stack(outs, dim=1)

    # ---- training / plain evaluation -------------------------------------
    def forward(self, x, labels=None, snr=None, ratio=None, chan=None, view_seed=None):
        cfg = self.cfg
        if self.training:
            snr = snr if snr is not None else torch.empty(1).uniform_(*cfg.train_snr).item()
            ratio = ratio if ratio is not None else random.choice(cfg.train_ratios)
            chan = chan or random.choice(cfg.train_channels)
        else:
            if snr is None:
                raise ValueError("pass snr explicitly at evaluation time")
            ratio = 1.0 if ratio is None else ratio
            chan = chan or cfg.train_channels[0]

        ws, ys = self._transmit(self._views(x, view_seed), snr, ratio, chan)
        logits, proj = self._decode(ys, [True] * self.K)
        out = {"logits": logits, "snr": snr, "ratio": ratio, "chan": chan}

        if labels is not None and self.training:
            loss_task = F.cross_entropy(logits, labels)                 # tau = K
            order = torch.randperm(self.K).tolist()                     # random UAV order
            for tau in range(1, self.K):                                # tau = 1..K-1
                keep = [False] * self.K
                for k in order[:tau]:
                    keep[k] = True
                loss_task = loss_task + F.cross_entropy(self._decode(ys, keep)[0], labels)

            loss_comp = torch.stack([w.abs().mean() for w in ws]).mean()
            loss_teacher = torch.zeros((), device=x.device)
            if self.use_teacher:
                z = self.clip(x)                                        # frozen, no grad
                if cfg.variant == "ecsc":
                    loss_teacher = contrastive_energy_loss(self.energy_net(z, proj), cfg.margin)
                else:
                    loss_teacher = (1.0 - F.cosine_similarity(proj, z, dim=-1)).mean()

            out["loss"] = loss_task + cfg.lam_comp * loss_comp + cfg.lam_teacher * loss_teacher
            out["parts"] = {"task": loss_task.item(), "comp": loss_comp.item(),
                            "teacher": float(loss_teacher.item())}
        return out
