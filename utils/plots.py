# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:41:55 2026

@author: Exaud
"""

""" `agg` maps (group, record_key) -> (mean, std, n_seeds) as built by make_tables.aggregate.
Pass `dataset` to put the dataset name in the figure title."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def _groups(agg):
    return sorted({g for g, _ in agg})


def label(g):
    ds, variant, K, view, train = g
    return f"{variant} (K={K}, {view}, train={train})"


def _title(text, dataset):
    return f"{dataset}: {text}" if dataset else text


def plot_acc_vs_snr(agg, chan, filename, ratio=1.0, dataset=None):
    plt.figure(figsize=(6, 4.5))
    for g in _groups(agg):
        pts = sorted((k[1], v) for (gg, k), v in agg.items() if gg == g and k[0] == chan and k[2] == ratio)
        if not pts:
            continue
        snr = [p[0] for p in pts]
        m = [100 * p[1][0][0] for p in pts]
        s = [100 * p[1][1][0] for p in pts]
        plt.errorbar(snr, m, yerr=s, marker="o", capsize=3, label=label(g))
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title(_title(f"{chan}, all UAVs transmit", dataset))
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_delta_sweep(agg, chan, snr, filename, ratio=1.0, dataset=None):
    """Accuracy vs average latency, one point per delta (early-exit trade-off)."""
    plt.figure(figsize=(6, 4.5))
    for g in _groups(agg):
        pts = sorted((k[3], v) for (gg, k), v in agg.items()
                     if gg == g and k[0] == chan and k[1] == snr and k[2] == ratio)
        if not pts:
            continue
        # metrics order: acc, rounds, bits, latency_ms
        plt.plot([p[1][0][3] for p in pts], [100 * p[1][0][0] for p in pts], marker="o", label=label(g))
    plt.xlabel("Average end-to-end latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title(_title(f"Early-exit sweep, {chan}, {snr:g} dB", dataset))
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
