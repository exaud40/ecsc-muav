# Energy-Calibrated Task-Oriented Communication for Cooperative Multi-UAV Edge Inference

**Authors:** Exaud Tarimo†‡, Huanlai Xing†‡*, Lexi Xu§, Zonghai Zhu† and Li Feng†  
†School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, China  
‡Engineering Research Center of Sustainable Urban Intelligent Transportation, Ministry of Education, Chengdu, China  
§Research Institute, China United Network Communications Corporation, Beijing, China  
**Corresponding Author:** Huanlai Xing  

*Code repository:* [ECSC-MUAV](https://github.com/exaud40/ecsc-muav)

---

## Overview

This repository provides the official implementation of **Energy-Calibrated Semantic Communication for Multi-UAV (ECSC-MUAV)**.

Multi-UAV systems are increasingly used for collaborative perception and real-time edge intelligence in next-generation wireless networks, but sending high-dimensional observations over bandwidth-limited, noisy channels is costly. ECSC-MUAV is a task-oriented semantic communication framework in which:

- each UAV encodes **its own view** with a lightweight encoder into a compact feature vector
- the features are sent over a noisy (AWGN, Rayleigh or Rician) channel and fused at an edge server for inference task
- the server stops as soon as its prediction is confident enough, so extra UAVs are asked to transmit only when needed (**confidence-based early exit**)
- during **training only**, a frozen CLIP teacher and an energy-based model (EnergyNet) calibrate the server's fused feature, which improves accuracy at low SNR without adding any cost at inference.

## Key Features

- **Lightweight onboard encoder.** Applied to the UAV's own view only and no CLIP on board.
- **Energy-based calibration (EnergyNet, training only).** A pairwise energy model aligns the server's projected feature with the CLIP embedding of the full image using a contrastive hinge loss. Neither CLIP nor EnergyNet is used at inference.
- **Confidence-based selective retransmission.** UAV 1 transmits first. After each round the server classifies from the UAVs received so far and stops if the maximum softmax probability is at least `delta`; otherwise the next UAV transmits. One stop/continue message is fed back between rounds.
- **Flexible transmission budget.** A random fraction `ratio` of the feature symbols is sent, so one model covers several bit budgets.
  

## Proposed Architecture

![System Architecture](figures/system_architecture.png)

1. Each UAV observes its own view (see [Views](#uav-views)).
2. Each UAV encodes its view and power-normalises the feature.
3. The feature is transmitted over the wireless channel.
4. The edge server decodes the received features (missing UAVs are zero-filled) and classifies.
5. Early exit: the server stops if confident, otherwise it requests the next UAV.
6. **Training only:** a frozen CLIP teacher sees the full image, and EnergyNet calibrates the server's projected feature against the CLIP embedding.

---

## Repository Structure

```
ecsc-muav/
├── main.py                  
├── config.py                                                      
├── requirements.txt
├── data/
│   └── dataloader.py        
├── models/
│   ├── net.py               
│   ├── encoder.py           
│   ├── decoder.py           
│   ├── channel.py           
│   ├── views.py             
│   ├── energy.py            
│   └── feature_extractor.py 
└── utils/
    ├── train.py 
    ├── common.py                   
    ├── costs.py
    ├── eval_all.py
    ├── make_tables.py              
    └── plots.py            
```

---

## Datasets

| `--dataset`     | Dataset            | Classes | Split used                                              |
|-----------------|--------------------|---------|---------------------------------------------------------|
| `uc_merced`     | UC Merced Land Use | 21      | stratified 70 / 10 / 20 (train / val / test)            |
| `rsscn`         | RSSCN7             | 7       | stratified 70 / 10 / 20 (train / val / test)            |
| `nwpu_resisc45` | NWPU-RESISC45      | 45      | provided `train/` and `test/`; 10 % of `train/` for val |

Download the datasets from their official sources and arrange them as below. These are the default locations; use `--data_root <path>` to put them anywhere else.

```
data/
├── UCMerced_LandUse/Images/<class_name>/*.tif        # uc_merced
├── RSSCN7/<class_name>/*.jpg                         # rsscn
└── NWPU-RESISC45/
    ├── train/<class_name>/*.jpg                      # nwpu_resisc45
    └── test/<class_name>/*.jpg
```

`uc_merced` and `rsscn` use a single folder with one sub-folder per class, split internally with `--split 0.7 0.1 0.2`. `nwpu_resisc45` needs `train/` and `test/` folders, and the validation set is carved out of `train/` (`--val_frac`, default 0.1). In all cases the split is fixed by `--split_seed` (default 0), so every seed and every variant is evaluated on exactly the same test images.

---

## Installation

Requires Python 3.10+ and PyTorch 2.1+.

```bash
git clone https://github.com/exaud40/ecsc-muav.git
cd ecsc-muav
pip install -r requirements.txt
```

`requirements.txt` installs `torch`, `torchvision`, `transformers`, `numpy`, `matplotlib`, `tqdm`, `pillow` and `thop`.

The first run with the `ecsc` or `cosine_distill` variants downloads the CLIP teacher (`openai/clip-vit-base-patch32`) from [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32).

---

## Usage

Run everything from the repository root. `--phase` selects what to run: `train`, `eval`, `tables`, or `all` (default).

### Quick start: full pipeline on one dataset

```bash
python main.py --dataset uc_merced --variant ecsc --seed 0
```

This trains the model, evaluates the best checkpoint on the test set over the full channel/SNR/bit-budget/threshold grid, and builds the tables and figures.

### Other datasets

```bash
python main.py --dataset rsscn         --data_root /path/to/RSSCN7
python main.py --dataset nwpu_resisc45 --data_root /path/to/NWPU-RESISC45
```

### Individual phases

```bash
python main.py --phase train  --dataset uc_merced --variant no_teacher --seed 1
python main.py --phase eval   --dataset uc_merced --variant no_teacher --seed 1
python main.py --phase tables
```

`eval` loads the checkpoint of the matching run, so it must use the same `--dataset`, `--variant`, `--K`, `--view_mode`, `--train_channels` and `--seed` as `train`.

### Model variants

| `--variant`      | Teacher term during training                                          |
|------------------|-----------------------------------------------------------------------|
| `ecsc` (default) | contrastive energy loss between the CLIP embedding and the projected fused feature (EnergyNet) |
| `cosine_distill` | control: `1 - cos(projection, CLIP embedding)`, no EnergyNet          |
| `no_teacher`     | none; CLIP is not loaded, so no download is needed                    |

### Reproducing the results

Run each variant over several seeds, then build the tables. Mean and standard deviation are computed over seeds.

```bash
for ds in uc_merced rsscn nwpu_resisc45; do
  for variant in ecsc cosine_distill no_teacher; do
    for seed in 0 1 2; do
      python main.py --phase all --dataset $ds --variant $variant --seed $seed
    done
  done
done
python main.py --phase tables
```

### Other options

```bash
# One model for all channels (training with fading)
python main.py --dataset uc_merced --train_channels awgn rayleigh rician

# Overlapping random-crop views and a different number of UAVs
python main.py --dataset uc_merced --view_mode overlap --K 4

# Windows or low-memory machines: avoid DataLoader worker problems
python main.py --dataset uc_merced --num_workers 0
```

Every option and its default is listed in [`config.py`](config.py). Run `python main.py --help` for the full list.

### Smoke test (not for reported results)

To check that the whole pipeline runs end to end:

```bash
python main.py --dataset rsscn --data_root <path> --epochs 1 \
    --eval_snrs 10 --eval_channels awgn --clip_name stub
```

> **Note: `--clip_name stub` is for smoke tests only.** It replaces CLIP with a tiny randomly initialised network, so the teacher carries no semantic information. The stub only affects those two variants; `no_teacher` does not use a teacher at all.

### Choosing the teacher for real experiments

For reported results, leave `--clip_name` at its default (`openai/clip-vit-base-patch32`, 512-d embedding). Another CLIP model can be used, as long as `--teacher_dim` equals that model's projection dimension (the code checks this):

```bash
# CLIP ViT-B/16 (512-d)
python main.py --dataset uc_merced --clip_name openai/clip-vit-base-patch16 --teacher_dim 512

# CLIP ViT-L/14 (768-d)
python main.py --dataset uc_merced --clip_name openai/clip-vit-large-patch14 --teacher_dim 768
```

A local folder with a downloaded CLIP checkpoint also works as `--clip_name`. To run without any teacher, use `--variant no_teacher`.

---

## Outputs

| Location                                  | Content                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------|
| `checkpoints/<run>.pt`                    | best checkpoint (selected on validation accuracy; CLIP weights excluded)|
| `results/logs/<run>_log.json`             | per-epoch training loss, training accuracy and validation accuracy      |
| `results/<run>.json`                      | full test results of one run, with the exact configuration, git commit and the early-exit threshold `delta*` chosen on the validation set |
| `results/tables/full.csv`, `early_exit.csv`, `complexity.json` | aggregated results over seeds, all datasets together |
| `results/tables/<dataset>/`               | accuracy-vs-SNR figures, early-exit trade-off figure and tables per dataset |

`<run>` is `<dataset>_<variant>_K<K>_<view_mode>_<train_channels>_seed<seed>`. Every number in the tables is built from the `results/*.json` files; nothing is typed by hand.

---
## Citation

If you find this work useful, please cite:

```bibtex
@article{ecsc_muav_2026,
  title={Energy-Calibrated Task-Oriented Communication for Cooperative Multi-UAV Edge Inference},
  author={Tarimo, Exaud and Xing, Huanlai and Xu, Lexi and Zhu, Zonghai and Feng, Li},
  journal={TBD},
  year={2026}
}
```

## Contact

For questions or collaborations: exaudtarimo@my.swjtu.edu.cn

## Acknowledgement

This work is supported by the **Sichuan Science and Technology Program (2024NSFSC1472)**.
