#!/usr/bin/env python
"""Benchmark de DEBIT fidele au step reel train_gmic (8 bits + gpu-augment).

Replique exactement : x.to(device).float() -> [GPUAugment] -> gpu_standardize
-> forward AMP -> loss 3 tetes -> backward -> grad-clip -> step. Utilise le VRAI
dataloader (decode PNG reel) -> mesure le bottleneck decode/transfert/GPU.
Chronometre apres warmup. 1 invocation = 1 config (pour boucler en shell)."""
import argparse
import os
import resource
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

ROOT = "/home/joshua/breast_cancer"
sys.path.insert(0, os.path.join(ROOT, "GMIC"))
sys.path.insert(0, ROOT)

from fine_tuning.dataset_gmic import make_loaders, INPUT_SIZE  # noqa
from fine_tuning.gmic_finetune import build_gmic, default_parameters, param_groups  # noqa
from fine_tuning.train_gmic import compute_loss, gpu_standardize, GPUAugment  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--gpu-augment", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--image-dir", default="data/preprocess_image/cropped_images")
    ap.add_argument("--weights", default="GMIC/models/sample_model_1.p")
    ap.add_argument("--percent-t", type=float, default=0.02)
    args = ap.parse_args()

    device = torch.device("cuda")
    train_loader, _, train_samples, _ = make_loaders(
        pkl_path="data/preprocess_image/data.pkl", image_dir=args.image_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        augment=not args.gpu_augment, input_size=INPUT_SIZE,
    )
    n_train = len(train_samples)
    steps_per_epoch = n_train // args.batch_size

    params = default_parameters(percent_t=args.percent_t, gpu_number=0)
    model = build_gmic(args.weights, params, device)
    model.train()
    optimizer = torch.optim.Adam(param_groups(model, 1e-5, 1e-4), weight_decay=1e-5)
    scaler = GradScaler("cuda")
    gpu_aug = GPUAugment(input_size=INPUT_SIZE).to(device) if args.gpu_augment else None

    torch.cuda.reset_peak_memory_stats()
    it = iter(train_loader)
    t_start = None
    done = 0
    target = args.warmup + args.steps
    while done < target:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x = x.to(device, non_blocking=True).float()
        if gpu_aug is not None:
            x = gpu_aug(x)
        x = gpu_standardize(x)
        yt = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            yf, yg, yl, sal = model(x)
        total, _ = compute_loss(yf, yg, yl, sal, yt, 1e-4)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        done += 1
        if done == args.warmup:
            torch.cuda.synchronize(); t_start = time.time()

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    sps = args.steps / elapsed
    imgs = sps * args.batch_size
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    epoch_min = steps_per_epoch / sps / 60.0
    maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB (main seul)
    aug = "gpu-aug" if args.gpu_augment else "cpu-aug"
    print(f"RESULT | batch={args.batch_size:>2} workers={args.num_workers} {aug:7s} "
          f"| {imgs:5.1f} img/s | {elapsed/args.steps*1000:5.0f} ms/step "
          f"| ~{epoch_min:4.1f} min/epoch | VRAM {peak_vram:4.1f} GB "
          f"| RSS_main {maxrss_gb:4.1f} GB", flush=True)


if __name__ == "__main__":
    main()
