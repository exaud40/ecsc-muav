"""Evaluates a trained checkpoint on the TEST split over the full grid and writes ONE
JSON file per run. All tables and figures are built from these files by make_tables.py;
no number in the paper should be typed by hand."""
import os
from dataclasses import asdict

import torch

from config import run_name
from data.dataloader import get_loaders
from models.net import MultiUAV
from utils.common import get_device, git_commit, save_json
from utils.costs import bits_per_uav, complexity_report, latency_ms
from utils.evaluate import accuracy, collect_logits, early_exit_from_logits


def select_delta(early, full, cfg):
    """Smallest delta whose validation accuracy is within delta_eps of full transmission."""
    def at_ref(r):
        return r["split"] == "val" and r["chan"] == cfg.ref_chan and r["snr"] == cfg.ref_snr and r["ratio"] == 1.0
    acc_full = [r["acc"] for r in full if at_ref(r)][0]
    ok = sorted(r["delta"] for r in early if at_ref(r) and r["acc"] >= acc_full - cfg.delta_eps)
    return ok[0] if ok else max(cfg.deltas)


def run_eval(cfg):
    device = get_device(cfg.device)
    loaders, num_classes = get_loaders(cfg)
    model = MultiUAV(cfg, num_classes).to(device)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{run_name(cfg)}.pt")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_trainable_state_dict(ck["model"])
    model.eval()

    comp = complexity_report(model, cfg, device)
    t_enc, t_srv = comp["t_enc_ms"] / 1e3, comp["t_srv_ms"] / 1e3

    combos = {(c, s, 1.0) for c in cfg.eval_channels for s in cfg.eval_snrs}
    combos |= {(cfg.ref_chan, cfg.ref_snr, r) for r in cfg.eval_ratios}   # bit-budget sweep at ref point
    combos.add((cfg.ref_chan, cfg.ref_snr, 1.0))

    full, early = [], []
    for split in ("val", "test"):
        for chan, snr, ratio in sorted(combos):
            if split == "val" and (chan, snr, ratio) != (cfg.ref_chan, cfg.ref_snr, 1.0):
                continue                                   # validation only needed to choose delta*
            logits, y = collect_logits(model, loaders[split], snr, ratio, chan, device, seed=cfg.seed)
            full.append({"split": split, "chan": chan, "snr": snr, "ratio": ratio,
                         "acc": accuracy(logits[:, -1].argmax(-1), y)})
            for delta in cfg.deltas:
                preds, rounds = early_exit_from_logits(logits, delta)
                lat = latency_ms(rounds, cfg, snr, ratio, t_enc, t_srv)
                early.append({"split": split, "chan": chan, "snr": snr, "ratio": ratio, "delta": delta,
                              "acc": accuracy(preds, y),
                              "rounds": rounds.float().mean().item(),
                              "bits": (rounds.float() * bits_per_uav(cfg, ratio)).mean().item(),
                              "latency_ms": lat.mean().item()})
            print(f"[eval] {split} {chan} snr={snr} ratio={ratio}: full acc {full[-1]['acc']:.4f}")

    result = {"meta": {**asdict(cfg), "commit": git_commit(), "num_classes": num_classes,
                       "n_test": len(loaders["test"].dataset), "best_val_epoch": ck["epoch"],
                       "best_val_acc": ck["val_acc"]},
              "delta_star": select_delta(early, full, cfg),
              "complexity": comp, "full": full, "early": early}
    out = os.path.join(cfg.results_dir, f"{run_name(cfg)}.json")
    save_json(result, out)
    print(f"[eval] delta* = {result['delta_star']} | wrote {out}")
    return out
