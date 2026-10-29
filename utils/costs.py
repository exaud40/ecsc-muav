# -*- coding: utf-8 -*-
"""
Created on Thu Oct 07 10:36:58 2026

@author: Exaud
"""

"""Bits, latency and complexity. Every assumption is a field of Config.

Latency of a sample that stops after `rounds` rounds (one UAV per round, each
UAV uses the full bandwidth W, one stop/continue message between rounds):

  T = t_enc + rounds * (T_up + T_prop + t_srv) + (rounds - 1) * (T_fb + T_prop)
  T_up = bits_per_uav / C,   T_fb = feedback_bits / C,   C = W log2(1 + psi)

C is the Shannon capacity at the AVERAGE SNR: an ideal capacity-achieving code,
no fading outage, no protocol overhead. t_enc and t_srv are measured on the
machine running the code"""

import copy
import math
import time

import torch

C_LIGHT = 3e8


def bits_per_uav(cfg, ratio):
    return cfg.bits_per_element * cfg.bottleneck_dim * ratio


def capacity_bps(cfg, snr_db):
    return cfg.bandwidth_hz * math.log2(1.0 + 10.0 ** (snr_db / 10.0))


def latency_ms(rounds, cfg, snr_db, ratio, t_enc_s, t_srv_s):
    """rounds: tensor of per-sample round counts. Returns per-sample latency in ms."""
    cap = capacity_bps(cfg, snr_db)
    t_up = bits_per_uav(cfg, ratio) / cap
    t_fb = cfg.feedback_bits / cap
    prop = cfg.distance_m / C_LIGHT
    r = rounds.float()
    return 1e3 * (t_enc_s + r * (t_up + prop + t_srv_s) + (r - 1.0) * (t_fb + prop))


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _macs(module, example):
    try:
        from thop import profile
        macs, _ = profile(copy.deepcopy(module), inputs=(example,), verbose=False)
        return float(macs)
    except Exception:
        return None   # install thop (pip install thop) to get MACs


def _time_ms(fn, n=100, warmup=10, device="cpu"):
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return 1e3 * (time.perf_counter() - t0) / n


@torch.no_grad()
def complexity_report(model, cfg, device):
    model.eval()
    hw = (256, max(1, 256 // cfg.K)) if cfg.view_mode == "patch" else (128, 128)
    view = torch.rand(1, 3, *hw, device=device)
    enc = model.encoders[0]
    y_cat = torch.zeros(1, cfg.K * cfg.bottleneck_dim, device=device)
    macs_enc = _macs(enc, view)
    rep = {
        "view_hw": list(hw),
        "params_encoder_per_uav": _count_params(enc),
        "macs_encoder_per_uav": macs_enc,
        "onboard_energy_estimate_J": None if macs_enc is None else macs_enc * cfg.joule_per_mac,
        "params_decoder": _count_params(model.decoder),
        "params_energy_net_training_only": _count_params(model.energy_net) if model.energy_net else 0,
        "params_clip_teacher_training_only": _count_params(model.clip) if model.clip else 0,
        "t_enc_ms": _time_ms(lambda: enc(view), device=device),
        "t_srv_ms": _time_ms(lambda: model.decoder(y_cat), device=device),
        "note": "times measured on the local device",
    }
    return rep
