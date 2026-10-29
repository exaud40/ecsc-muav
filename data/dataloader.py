# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 10:34:30 2026

@author: Exaud
"""

import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def stratified_split(targets, fractions, seed):
    """Per-class split into train/val/test according to `fractions` (ratio mode)."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    tr, va, te = [], [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.where(targets == c)[0])
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        tr += idx[:n_tr].tolist()
        va += idx[n_tr:n_tr + n_va].tolist()
        te += idx[n_tr + n_va:].tolist()
    return np.array(tr), np.array(va), np.array(te)


def stratified_val_split(targets, val_frac, seed):
    """Per-class split of the training set into train/val (predefined mode)."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    tr, va = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.where(targets == c)[0])
        n_va = int(round(val_frac * len(idx)))
        va += idx[:n_va].tolist()
        tr += idx[n_va:].tolist()
    return np.array(tr), np.array(va)


def _transforms():
    train_tf = transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])
    eval_tf = transforms.Compose([transforms.Resize((256, 256)),
                                  transforms.ToTensor()])
    return train_tf, eval_tf


def _check_dir(path, cfg):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset folder not found: '{path}'\n"
            f"Download the '{cfg.dataset}' dataset and either place it there "
            f"or pass --data_root <path>.")


def _loaders_ratio(cfg, train_tf, eval_tf):
    _check_dir(cfg.data_root, cfg)
    ds_train = ImageFolder(cfg.data_root, train_tf)   # augmentation only for training
    ds_eval = ImageFolder(cfg.data_root, eval_tf)
    tr, va, te = stratified_split(ds_eval.targets, cfg.split, cfg.split_seed)
    parts = {"train": (ds_train, tr), "val": (ds_eval, va), "test": (ds_eval, te)}
    return parts, ds_eval.classes


def _loaders_predefined(cfg, train_tf, eval_tf):
    train_dir = os.path.join(cfg.data_root, "train")
    test_dir = os.path.join(cfg.data_root, "test")
    _check_dir(train_dir, cfg)
    _check_dir(test_dir, cfg)
    ds_train = ImageFolder(train_dir, train_tf)       # augmentation: training only
    ds_val = ImageFolder(train_dir, eval_tf)          # same files, no augmentation
    ds_test = ImageFolder(test_dir, eval_tf)
    assert ds_train.classes == ds_test.classes, "train/ and test/ class folders differ"
    tr, va = stratified_val_split(ds_val.targets, cfg.val_frac, cfg.split_seed)
    parts = {"train": (ds_train, tr), "val": (ds_val, va), "test": (ds_test, None)}
    return parts, ds_train.classes


def get_loaders(cfg):
    """Returns ({'train','val','test'}: DataLoader, num_classes)."""
    train_tf, eval_tf = _transforms()
    if cfg.split_mode == "ratio":
        parts, classes = _loaders_ratio(cfg, train_tf, eval_tf)
    elif cfg.split_mode == "predefined":
        parts, classes = _loaders_predefined(cfg, train_tf, eval_tf)
    else:
        raise ValueError(f"Unknown split_mode '{cfg.split_mode}'")

    def mk(ds, ids, shuffle):
        if ids is not None:
            ds = Subset(ds, ids)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, drop_last=shuffle)

    loaders = {k: mk(ds, ids, shuffle=(k == "train")) for k, (ds, ids) in parts.items()}
    sizes = {k: len(l.dataset) for k, l in loaders.items()}
    print(f"[data] {cfg.dataset} ({cfg.split_mode}) | {len(classes)} classes | "
          f"train {sizes['train']} | val {sizes['val']} | test {sizes['test']}")
    return loaders, len(classes)
