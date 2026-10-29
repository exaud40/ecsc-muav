# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:40:58 2026

@author: Exaud
"""


import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from utils.plots import label, plot_acc_vs_snr, plot_delta_sweep


def load_runs(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def group_of(meta):
    return (meta["dataset"], meta["variant"], meta["K"], meta["view_mode"], "-".join(meta["train_channels"]))


def aggregate(runs, section, split, rec_keys, metrics):
    buckets = defaultdict(list)
    for r in runs:
        g = group_of(r["meta"])
        for rec in r[section]:
            if rec["split"] == split:
                buckets[(g, tuple(rec[k] for k in rec_keys))].append([rec[m] for m in metrics])
    return {k: (np.mean(v, 0), np.std(v, 0), len(v)) for k, v in buckets.items()}


def for_dataset(agg, dataset):
    """Keep only the entries of one dataset (the group tuple starts with the dataset name)."""
    return {k: v for k, v in agg.items() if k[0][0] == dataset}


def write_csv(agg, rec_keys, metrics, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "K", "view", "train_channels", *rec_keys,
                    *[f"{m}_mean" for m in metrics], *[f"{m}_std" for m in metrics], "n_seeds"])
        for (g, k), (mu, sd, n) in sorted(agg.items(), key=lambda t: str(t[0])):
            w.writerow([*g, *k, *mu, *sd, n])


def latex_acc_table(agg, chan, snrs, path):
    lines = ["\\begin{tabular}{l" + "c" * len(snrs) + "}", "\\hline",
             "Method & " + " & ".join(f"{s:g} dB" for s in snrs) + " \\\\", "\\hline"]
    for g in sorted({g for g, _ in agg}):
        cells = []
        for s in snrs:
            v = agg.get((g, (chan, s, 1.0)))
            cells.append("--" if v is None else f"{100 * v[0][0]:.2f} $\\pm$ {100 * v[1][0]:.2f}")
        lines.append(label(g).replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_all(cfg):
    runs = load_runs(cfg.results_dir)
    if not runs:
        print("[tables] no result files found")
        return
    out = os.path.join(cfg.results_dir, "tables")
    os.makedirs(out, exist_ok=True)

    full = aggregate(runs, "full", "test", ("chan", "snr", "ratio"), ("acc",))
    early = aggregate(runs, "early", "test", ("chan", "snr", "ratio", "delta"),
                      ("acc", "rounds", "bits", "latency_ms"))
    write_csv(full, ("chan", "snr", "ratio"), ("acc",), os.path.join(out, "full.csv"))
    write_csv(early, ("chan", "snr", "ratio", "delta"), ("acc", "rounds", "bits", "latency_ms"),
              os.path.join(out, "early_exit.csv"))

    # one set of figures / LaTeX tables per dataset, so datasets are never mixed in one plot
    for ds in sorted({r["meta"]["dataset"] for r in runs}):
        ds_dir = os.path.join(out, ds)
        os.makedirs(ds_dir, exist_ok=True)
        full_ds, early_ds = for_dataset(full, ds), for_dataset(early, ds)
        for chan in sorted({k[0] for _, k in full_ds}):
            plot_acc_vs_snr(full_ds, chan, os.path.join(ds_dir, f"acc_vs_snr_{chan}.pdf"), dataset=ds)
            latex_acc_table(full_ds, chan, cfg.eval_snrs, os.path.join(ds_dir, f"acc_table_{chan}.tex"))
        plot_delta_sweep(early_ds, cfg.ref_chan, cfg.ref_snr,
                         os.path.join(ds_dir, "delta_sweep.pdf"), dataset=ds)

    comp = {group_of(r["meta"]) + (r["meta"]["seed"],): r["complexity"] for r in runs}
    with open(os.path.join(out, "complexity.json"), "w") as f:
        json.dump({str(k): v for k, v in comp.items()}, f, indent=2)
    print(f"[tables] {len(runs)} run(s), {len({r['meta']['dataset'] for r in runs})} dataset(s) -> {out}")
