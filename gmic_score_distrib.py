#!/usr/bin/env python
"""
Distribution des scores predits par GMIC sur le set de VALIDATION, avec best.pt
(epoch du meilleur AUC_mal). Deux panneaux : tete malignant et tete benign.
Dans chaque panneau, distribution des scores des VRAIS positifs vs VRAIS negatifs,
pour visualiser la separation des classes / choisir un seuil.

Sortie : scores npz + figure PNG dans fine_tuning/checkpoints/gmic_ft/.
N'utilise QUE le forward (read-only) -> peut tourner pendant l'entrainement.
"""
import os
import sys

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# evite l'epuisement de file descriptors avec les workers (RuntimeError "Too many open files")
torch.multiprocessing.set_sharing_strategy("file_system")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GMIC"))
sys.path.insert(0, PROJECT_ROOT)

from fine_tuning.dataset_gmic import (  # noqa: E402
    load_and_split3, build_samples, GMICViewDataset, INPUT_SIZE,
)
from fine_tuning.gmic_finetune import build_gmic, default_parameters  # noqa: E402

OUT = "fine_tuning/checkpoints/gmic_ft"
PKL = "data/preprocess_image/data.pkl"
IMG_DIR = "data/preprocess_image/cropped_images"
CKPT = os.path.join(OUT, "best.pt")


def gpu_standardize(x):
    x = x.float()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp(min=1e-5)
    return (x - m) / s


@torch.no_grad()
def main():
    device = torch.device("cuda")

    # --- meme split que l'entrainement (seed 42, fracs par defaut) ---
    _, _, val_ex, _ = load_and_split3(PKL)
    val_samples = build_samples(val_ex, IMG_DIR)
    val_ds = GMICViewDataset(val_samples, augment=False, input_size=INPUT_SIZE,
                             to_uint8=True)
    # entrainement termine -> GPU libre. workers=0 : single-process, zero souci de fd.
    loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=False)

    # --- modele : best.pt ---
    params = default_parameters(percent_t=0.02, gpu_number=0)
    model = build_gmic(None, params, device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[ckpt] best.pt epoch={ck.get('epoch')} best_auc={ck.get('best_auc'):.4f}")

    pb, pm, lab = [], [], []
    for i, (x, y) in enumerate(loader):
        x = gpu_standardize(x.to(device, non_blocking=True))
        with autocast("cuda"):
            yf, yg, yl, sal = model(x)
        yf = yf.float().cpu().numpy()
        pb.extend(yf[:, 0].tolist())
        pm.extend(yf[:, 1].tolist())
        lab.append(y.numpy())
        if i % 25 == 0:
            print(f"  batch {i} ({len(pm)} vues)")
    pb = np.array(pb)
    pm = np.array(pm)
    lab = np.concatenate(lab)
    lb, lm = lab[:, 0].astype(int), lab[:, 1].astype(int)

    auc_m = roc_auc_score(lm, pm)
    auc_b = roc_auc_score(lb, pb)
    print(f"[check] AUC_mal={auc_m:.4f} (n_pos={lm.sum()}/{len(lm)}) | "
          f"AUC_ben={auc_b:.4f} (n_pos={lb.sum()}/{len(lb)})")

    np.savez(os.path.join(OUT, "val_scores.npz"),
             p_benign=pb, p_malignant=pm, lab_benign=lb, lab_malignant=lm,
             epoch=ck.get("epoch"))

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    def panel(ax, score, y, head, auc):
        pos, neg = score[y == 1], score[y == 0]
        ax.hist(neg, bins=bins, density=True, alpha=0.55, color="#3b7dd8",
                label=f"negatifs (n={len(neg)})")
        ax.hist(pos, bins=bins, density=True, alpha=0.6, color="#d83b3b",
                label=f"positifs (n={len(pos)})")
        ax.set_title(f"Tete {head}  |  AUC={auc:.3f}")
        ax.set_xlabel(f"score predit  P({head})")
        ax.set_ylabel("densite")
        ax.legend()
        ax.grid(alpha=0.25)

    panel(axes[0], pm, lm, "malignant", auc_m)
    panel(axes[1], pb, lb, "benign", auc_b)
    fig.suptitle(f"Distribution des scores GMIC sur la validation "
                 f"(best.pt, epoch {ck.get('epoch')})", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(OUT, "val_score_distrib.png")
    fig.savefig(out_png, dpi=120)
    print(f"[ok] figure -> {out_png}")


if __name__ == "__main__":
    main()
