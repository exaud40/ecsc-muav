# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 10:26:00 2026

@author: Exaud
"""

import argparse
from dataclasses import dataclass, field, fields
from typing import List


DATASET_PRESETS = {
    "uc_merced": dict(data_root="./data/UCMerced_LandUse/Images", split_mode="ratio"),
    "rsscn": dict(data_root="./data/RSSCN7", split_mode="ratio"),
    "nwpu_resisc45": dict(data_root="./data/NWPU-RESISC45", split_mode="predefined"),
}


@dataclass
class Config:
    phase: str = "all"                 # option may be  train | eval | tables | all
    results_dir: str = "./results"
    ckpt_dir: str = "./checkpoints"
    device: str = "auto"

    dataset: str = "uc_merced"         # uc_merced | rsscn | nwpu_resisc45 
    data_root: str = ""                
    split_mode: str = ""               
    split: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])  # train/val/test (ratio mode only)
    val_frac: float = 0.1              # fraction of train/ held out for validation (predefined mode only)
    split_seed: int = 0                
    batch_size: int = 32
    num_workers: int = 4

    K: int = 2                         # number of UAVs
    view_mode: str = "overlap"           # choose patch | overlap (see our models/views.py)
    bottleneck_dim: int = 32           # d' (must be even >> real features are paired into complex symbols)
    variant: str = "ecsc"              # option include  ecsc | cosine_distill | no_teacher
    clip_name: str = "openai/clip-vit-base-patch32"   
    teacher_dim: int = 512
    rician_k_lin: float = 10.0         # Rician K-factor, linear scale

    seed: int = 0
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lam_teacher: float = 0.1           # lambda_1 (energy / distillation term)
    lam_comp: float = 0.01             # lambda_2 (l1 sparsity on transmitted features)
    margin: float = 0.5                # margin m of the contrastive energy loss
    train_snr: List[float] = field(default_factory=lambda: [0.0, 20.0])      # U(a, b) dB
    train_ratios: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    train_channels: List[str] = field(default_factory=lambda: ["awgn"])      # sampled per batch

    eval_channels: List[str] = field(default_factory=lambda: ["awgn", "rayleigh", "rician"])
    eval_snrs: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0])
    eval_ratios: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])  
    deltas: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01])  # 1.01 = never exit early
    ref_snr: float = 10.0              # point used to choose delta* on the validation set
    ref_chan: str = "awgn"
    delta_eps: float = 0.01            # allowed val-accuracy drop of early exit vs full transmission

    
    bits_per_element: int = 16         # bit depth assumed for each transmitted real feature element
    bandwidth_hz: float = 2e5          # per-link bandwidth W
    feedback_bits: int = 8             # stop/continue message size
    distance_m: float = 1000.0         # only enters through propagation delay
    joule_per_mac: float = 5e-12       # estimated onboard energy per MAC

    def __post_init__(self):
        if self.dataset not in DATASET_PRESETS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. "
                             f"Choose from {sorted(DATASET_PRESETS)}.")
        preset = DATASET_PRESETS[self.dataset]
        if not self.data_root:
            self.data_root = preset["data_root"]
        if not self.split_mode:
            self.split_mode = preset["split_mode"]


def add_args(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    for f in fields(Config):
        v = getattr(defaults, f.name)
        if f.name == "dataset":
            parser.add_argument("--dataset", type=str, default=v, choices=sorted(DATASET_PRESETS))
        elif f.name in ("data_root", "split_mode"):
            # Empty default -> resolved per dataset in Config.__post_init__.
            parser.add_argument(f"--{f.name}", type=str, default="")
        elif isinstance(v, list):
            parser.add_argument(f"--{f.name}", nargs="+", type=type(v[0]), default=v)
        else:
            parser.add_argument(f"--{f.name}", type=type(v), default=v)


def from_args(args: argparse.Namespace) -> Config:
    cfg = Config(**vars(args))
    assert cfg.bottleneck_dim % 2 == 0, "bottleneck_dim must be even"
    assert cfg.variant in ("ecsc", "cosine_distill", "no_teacher")
    assert cfg.view_mode in ("patch", "overlap")
    assert cfg.split_mode in ("ratio", "predefined"), "split_mode must be ratio or predefined"
    if cfg.split_mode == "ratio":
        assert len(cfg.split) == 3 and abs(sum(cfg.split) - 1.0) < 1e-6, \
            "split must have 3 entries (train/val/test) summing to 1"
    else:
        assert 0.0 < cfg.val_frac < 1.0, "val_frac must be in (0, 1)"
    return cfg


def run_name(cfg: Config) -> str:
    return (f"{cfg.dataset}_{cfg.variant}_K{cfg.K}_{cfg.view_mode}_"
            f"{'-'.join(cfg.train_channels)}_seed{cfg.seed}")
