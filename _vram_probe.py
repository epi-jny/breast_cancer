#!/usr/bin/env python
"""Sonde VRAM fidele : replique le step reel train_gmic (float -> gpu_aug ->
gpu_standardize -> forward AMP -> loss 3 tetes -> backward -> grad-clip -> step)
sur un batch synthetique. 1 invocation = 1 batch (isole les OOM)."""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, os.path.join(ROOT, "GMIC"))
sys.path.insert(0, ROOT)

from fine_tuning.dataset_gmic import INPUT_SIZE  # noqa
from fine_tuning.gmic_finetune import build_gmic, default_parameters, param_groups  # noqa
from fine_tuning.train_gmic import compute_loss, gpu_standardize, GPUAugment  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p")
    ap.add_argument("--percent-t", type=float, default=0.02)
    args = ap.parse_args()

    device = torch.device("cuda")
    H, W = INPUT_SIZE
    B = args.batch_size
    try:
        params = default_parameters(percent_t=args.percent_t, gpu_number=0)
        model = build_gmic(args.weights, params, device)
        model.train()
        opt = torch.optim.Adam(param_groups(model, 1e-5, 1e-4), weight_decay=1e-5)
        scaler = GradScaler("cuda")
        gpu_aug = GPUAugment(input_size=INPUT_SIZE).to(device)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.steps):
            x = torch.randint(0, 4096, (B, 1, H, W), dtype=torch.float32, device=device)
            x = gpu_aug(x)
            x = gpu_standardize(x)
            y = torch.randint(0, 2, (B, 2), dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=True):
                yf, yg, yl, sal = model(x)
            total, _ = compute_loss(yf, yg, yl, sal, y, 1e-4)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"RESULT batch={B:>3} | VRAM alloc={alloc:5.1f} GB | "
              f"reserved={reserved:5.1f} GB / 45 GB | OK")
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        if "out of memory" in msg.lower():
            print(f"RESULT batch={B:>3} | OOM")
        else:
            print(f"RESULT batch={B:>3} | ERREUR: {msg}")


if __name__ == "__main__":
    main()
