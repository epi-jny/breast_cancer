#!/usr/bin/env python
"""Eval d'ENSEMBLE par sein (per-breast) sur la val canonique (seed 42, 1/6).

Pour chaque checkpoint : inference des probas malignant par vue (avec SON
percent_t propre), AUC solo par sein. Puis ensemble = MOYENNE des probas par vue
sur les modeles choisis (cf GMIC run_model.py qui moyenne benign/malignant_pred),
agregation par sein, AUC. + selection greedy forward.

IMPORTANT : --percent-ts donne le percent_t de CHAQUE checkpoint (meme ordre que
--ckpts). Les 5 modeles NYU ont des percent_t differents (0.02/0.03/0.03/0.05/0.10)
-> il FAUT inferer chacun avec le sien (le percent_t change le forward, top-t%)."""
import argparse
import sys
import os

import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, os.path.join(ROOT, "GMIC"))
sys.path.insert(0, ROOT)

from fine_tuning.dataset_gmic import make_loaders, INPUT_SIZE  # noqa
from fine_tuning.gmic_finetune import build_gmic, default_parameters  # noqa
from fine_tuning.train_gmic import gpu_standardize, _side  # noqa


@torch.no_grad()
def infer_probs(model, loader, device):
    """Probas malignant (index 1) de y_fusion, par vue, dans l'ordre du loader."""
    model.eval()
    out = []
    for x, _ in loader:
        x = gpu_standardize(x.to(device, non_blocking=True))
        with autocast("cuda"):
            yf, yg, yl, sal = model(x)
        out.extend(yf[:, 1].float().cpu().tolist())
    return np.array(out)


def per_breast(preds_view, samples):
    """Agrege les probas par vue en (pid, cote) -> proba moyenne + label."""
    from collections import defaultdict
    g, ym = defaultdict(list), {}
    for i, s in enumerate(samples):
        key = (s["pid"], _side(s["view"]))
        g[key].append(preds_view[i])
        ym[key] = int(s["label"][1])
    keys = list(g.keys())
    p = np.array([float(np.mean(g[k])) for k in keys])
    y = np.array([ym[k] for k in keys])
    return p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="chemins best.pt")
    ap.add_argument("--names", nargs="+", required=True, help="noms courts (meme ordre)")
    ap.add_argument("--percent-ts", nargs="+", type=float, default=None,
                    help="percent_t de CHAQUE ckpt (meme ordre). Defaut: 0.02 partout.")
    ap.add_argument("--image-dir", default="data/preprocess_image/cropped_images")
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p",
                    help="archi de base (ecrasee par le state_dict du ckpt) ; le percent_t seul compte")
    args = ap.parse_args()
    assert len(args.ckpts) == len(args.names), "ckpts et names de tailles differentes"
    pts = args.percent_ts if args.percent_ts is not None else [0.02] * len(args.ckpts)
    assert len(pts) == len(args.ckpts), "percent-ts doit avoir autant d'elements que ckpts"

    device = torch.device("cuda")
    # val canonique : memes fracs/seed que tous les runs -> val identique
    _, val_loader, _, val_samples = make_loaders(
        pkl_path="data/preprocess_image/data.pkl", image_dir=args.image_dir,
        batch_size=16, num_workers=6, augment=False, input_size=INPUT_SIZE,
    )

    view_probs = {}
    solo = {}
    print("\n=== AUC SOLO (per-breast) ===")
    for path, name, pt in zip(args.ckpts, args.names, pts):
        params = default_parameters(percent_t=pt, gpu_number=0)  # percent_t PROPRE
        model = build_gmic(args.weights, params, device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        pv = infer_probs(model, val_loader, device)
        view_probs[name] = pv
        p, y = per_breast(pv, val_samples)
        solo[name] = roc_auc_score(y, p)
        print(f"  {name:18s} (pt={pt:.2f}) : {solo[name]:.4f}")
        del model; torch.cuda.empty_cache()

    _, y = per_breast(view_probs[args.names[0]], val_samples)

    def ens_auc(names):
        pv = np.mean([view_probs[n] for n in names], axis=0)
        p, _ = per_breast(pv, val_samples)
        return roc_auc_score(y, p)

    print("\n=== ENSEMBLE de TOUS ===")
    print(f"  {'+'.join(args.names)} : {ens_auc(args.names):.4f}")

    order = sorted(args.names, key=lambda n: -solo[n])
    chosen = [order[0]]
    best = solo[order[0]]
    print("\n=== SELECTION GREEDY (ajout si amelioration) ===")
    print(f"  start {order[0]:18s} : {best:.4f}")
    for n in order[1:]:
        a = ens_auc(chosen + [n])
        if a > best + 1e-4:
            chosen = chosen + [n]; best = a
            print(f"  + {n:18s} -> {a:.4f}  (gardé)")
        else:
            print(f"  + {n:18s} -> {a:.4f}  (rejeté)")
    print(f"\n>>> MEILLEUR ENSEMBLE : {'+'.join(chosen)}"
          f"\n>>> AUC per-breast = {best:.4f}  (vs meilleur solo {max(solo.values()):.4f})")


if __name__ == "__main__":
    main()
