# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 10:22:58 2026

@author: Exaud
"""
import argparse

from config import add_args, from_args


def main():
    parser = argparse.ArgumentParser(description="ECSC-MUAV")
    add_args(parser)
    cfg = from_args(parser.parse_args())

    if cfg.phase in ("train", "all"):
        from utils.train import train
        train(cfg)
    if cfg.phase in ("eval", "all"):
        from utils.eval_all import run_eval
        run_eval(cfg)
    if cfg.phase in ("tables", "all"):
        from utils.make_tables import make_all
        make_all(cfg)


if __name__ == "__main__":     
    main()
