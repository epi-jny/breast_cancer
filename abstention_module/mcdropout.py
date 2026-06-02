import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, ConcatDataset, WeightedRandomSampler
import math
import scipy.special
import random as rd
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights
from typing import Tuple
from tqdm import tqdm
import pickle
import torch.optim.lr_scheduler as lr_scheduler


@torch.no_grad()
def mc_var_for_deployed_class(model, loader, device, T=30):
    """
    Step 1: Do a *deterministic* pass (model.eval()) to get the deployed prediction y_pred_det.
    Step 2: Enable dropout (model.train()) and run T stochastic passes.
    Step 3: For each sample, gather the probability of the *deterministic* class across passes,
            then return its mean and variance.

    Returns:
        y_pred_det         [N]   int64, class predicted by the deployed model
        mean_prob_detcls   [N]   mean P_t(y = y_pred_det | x) over T passes
        var_prob_detcls    [N]   variance of that probability over T passes
    """
    model = model.to(device)

    # ---- Step 1: deterministic deployed predictions (dropout OFF) ----
    model.eval()
    pred_list = []
    for xb, _ in tqdm(loader, desc="Deterministic pass"):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)                    # [B, C]
        pred_list.append(logits.softmax(dim=1).argmax(dim=1).cpu())
    y_pred_det = torch.cat(pred_list, dim=0)  # [N]

    # ---- Step 2: MC passes with dropout ON ----
    # If your model had BatchNorm, you'd want eval()+manually set dropout.train().
    model.train()  # your SmallCNN has no BN, so this is fine.

    probs_cols = []  # list of [N] tensors: prob of the deterministic class per pass
    offset = 0
    for t in range(T):
        batch_probs = []
        offset = 0
        for xb, _ in tqdm(loader, desc=f"MC pass {t+1}/{T}", leave=False):
            xb = xb.to(device, non_blocking=True)
            bsz = xb.shape[0]
            logits = model(xb)                    # [B, C]
            probs  = logits.softmax(dim=1)        # [B, C]
            idx = y_pred_det[offset:offset+bsz].to(xb.device)  # [B]
            p_det = probs.gather(1, idx.view(-1,1)).squeeze(1) # [B]
            batch_probs.append(p_det.cpu())
            offset += bsz
        probs_cols.append(torch.cat(batch_probs, dim=0))       # [N]

    mat = torch.stack(probs_cols, dim=1)        # [N, T]
    mean_prob_detcls = mat.mean(dim=1)          # [N]
    var_prob_detcls  = mat.var(dim=1, unbiased=False)  # [N]

    return y_pred_det, mean_prob_detcls, var_prob_detcls


def selective_accuracy(conf, y_hat, y_true, coverages=np.linspace(0.1, 1.0, 10)):
    """
    Compute accuracy at different coverage levels.

    Predictions are sorted by descending confidence, and accuracy is measured
    on the top fraction of samples defined by each coverage value.

    Args:
        conf (torch.Tensor): Confidence scores, shape (N,).
        y_hat (torch.Tensor): Predicted labels, shape (N,).
        y_true (torch.Tensor): True labels, shape (N,).
        coverages (array-like, optional): Fractions of data to keep (default: 0.1–1.0).

    Returns:
        (np.ndarray, np.ndarray): Coverages and corresponding accuracies.
    """
    order = torch.argsort(conf, descending=True)
    conf, y_hat, y_true = conf[order], y_hat[order], y_true[order]
    N = len(y_true)
    accs = []
    for c in coverages:
        k = max(1, int(c * N))
        accs.append((y_hat[:k] == y_true[:k]).float().mean().item())
    return np.array(coverages), np.array(accs)
