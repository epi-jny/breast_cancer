#!/usr/bin/env python
"""Ensemble PONDERE par sein. Infere chaque ckpt UNE fois (percent_t propre),
puis essaie plusieurs schemas de ponderation sur la moyenne des probas par vue.

Schemas testes :
 - equal        : moyenne equiponderee (baseline)
 - auc          : poids proportionnels a l'AUC solo per-breast
 - auc_excess   : poids proportionnels a (AUC_solo - 0.5)   [plus agressif]
 - softmax_auc  : softmax(AUC/T) sur les AUC solo (T=0.02)
 - drop_weak    : equiponderee SANS les modeles dont AUC_solo < seuil (def 0.86)
 - top2 / top3  : equiponderee sur les 2/3 meilleurs solo

NB : ponderer par l'AUC mesuree sur LA val = legerement optimiste (la val sert a
la fois a choisir les poids et a evaluer). C'est une borne haute douce ; le signal
robuste = est-ce que ponderer corrige la dilution par m2/m5."""
import argparse, sys, os
import numpy as np, torch
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, os.path.join(ROOT, "GMIC")); sys.path.insert(0, ROOT)
from fine_tuning.dataset_gmic import make_loaders, INPUT_SIZE
from fine_tuning.gmic_finetune import build_gmic, default_parameters
from fine_tuning.train_gmic import gpu_standardize, _side


@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval(); out = []
    for x, _ in loader:
        x = gpu_standardize(x.to(device, non_blocking=True))
        with autocast("cuda"):
            yf, yg, yl, sal = model(x)
        out.extend(yf[:, 1].float().cpu().tolist())
    return np.array(out)


def per_breast(preds_view, samples):
    from collections import defaultdict
    g, ym = defaultdict(list), {}
    for i, s in enumerate(samples):
        key = (s["pid"], _side(s["view"]))
        g[key].append(preds_view[i]); ym[key] = int(s["label"][1])
    keys = list(g.keys())
    return (np.array([float(np.mean(g[k])) for k in keys]),
            np.array([ym[k] for k in keys]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--percent-ts", nargs="+", type=float, required=True)
    ap.add_argument("--image-dir", default="data/preprocess_image/cropped_images")
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p")
    ap.add_argument("--drop-thresh", type=float, default=0.86)
    args = ap.parse_args()

    device = torch.device("cuda")
    _, val_loader, _, val_samples = make_loaders(
        pkl_path="data/preprocess_image/data.pkl", image_dir=args.image_dir,
        batch_size=16, num_workers=6, augment=False, input_size=INPUT_SIZE)

    names = args.names
    view_probs, solo = {}, {}
    print("\n=== AUC SOLO (per-breast) ===")
    for path, name, pt in zip(args.ckpts, names, args.percent_ts):
        params = default_parameters(percent_t=pt, gpu_number=0)
        model = build_gmic(args.weights, params, device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        view_probs[name] = infer_probs(model, val_loader, device)
        p, y = per_breast(view_probs[name], val_samples)
        solo[name] = roc_auc_score(y, p)
        print(f"  {name:10s} (pt={pt:.2f}) : {solo[name]:.4f}")
        del model; torch.cuda.empty_cache()

    _, y = per_breast(view_probs[names[0]], val_samples)

    def weighted_auc(sel_names, w):
        w = np.asarray(w, float); w = w / w.sum()
        pv = np.zeros_like(view_probs[sel_names[0]])
        for n, wi in zip(sel_names, w):
            pv += wi * view_probs[n]
        p, _ = per_breast(pv, val_samples)
        return roc_auc_score(y, p)

    aucs = np.array([solo[n] for n in names])
    schemes = {}
    schemes["equal"]       = (names, np.ones(len(names)))
    schemes["auc"]         = (names, aucs.copy())
    schemes["auc_excess"]  = (names, np.clip(aucs - 0.5, 1e-6, None))
    schemes["softmax_auc"] = (names, np.exp(aucs / 0.02))
    keep = [n for n in names if solo[n] >= args.drop_thresh]
    schemes[f"drop_weak(>={args.drop_thresh})"] = (keep, np.ones(len(keep))) if keep else (names, np.ones(len(names)))
    order = sorted(names, key=lambda n: -solo[n])
    schemes["top2"] = (order[:2], np.ones(2))
    schemes["top3"] = (order[:3], np.ones(3))

    print("\n=== ENSEMBLE PONDERE (per-breast) ===")
    print(f"  {'best solo':28s} : {max(solo.values()):.4f}  ({order[0]})")
    res = []
    for sname, (sel, w) in schemes.items():
        a = weighted_auc(sel, w)
        res.append((sname, a, sel))
        print(f"  {sname:28s} : {a:.4f}   [{'+'.join(sel)}]")
    best = max(res, key=lambda t: t[1])
    print(f"\n>>> MEILLEUR SCHEMA : {best[0]}  AUC={best[1]:.4f}  [{'+'.join(best[2])}]"
          f"\n>>> vs best solo {max(solo.values()):.4f}  /  moyenne equiponderee {dict((r[0],r[1]) for r in res)['equal']:.4f}")


if __name__ == "__main__":
    main()
