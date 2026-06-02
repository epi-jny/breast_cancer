import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import math
import scipy.special
from scipy.stats import beta
import random as rd
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights
from tqdm import tqdm
import pickle
import torch.optim.lr_scheduler as lr_scheduler
from scipy.special import gammaln
from collections import defaultdict
from pathlib import Path
from collections import Counter


def sotfmax(x):
    """
    x a vector of floats, returns softmaxes for these points
    """
    return np.exp(x) / np.exp(x).sum()


def binomial_log(n, j):
    """
    Return the log of the binomial coefficient "n choose j".

    Computed as:
        log(n! / (j! * (n - j)!))
      = gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)

    Parameters
    ----------
    n : int
        Total number of trials.
    j : int
        Number of successes.

    Returns
    -------
    float
        Logarithm of the binomial coefficient.
    """
    return gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)


def binom_sum(b, e, n):
    """
    binomial sum of term b in [0,1]
    with binomial coefs "j among n"
    for j in 0,1,...,e
    it is the proba of doing at most e errors among n Bernoulli iid experiences with error proba b
    """
    if e < n:
        v = np.array(
            [
                np.exp(binomial_log(n, j) + j * np.log(b) + (n - j) * np.log(1 - b))
                for j in range(e + 1)
            ]
        )
        return np.sum(v)
    elif e == n:
        return 1
    else:
        raise ValueError


def B_star(delta, e, n, b1=0, b2=1):
    """
    b_star recursive dichotomy
    approximate solution up to eps
    """
    eps = min(delta / 10, 1e-5)

    if (e == n) or (n == 0):
        return 1
    if e == 0:
        return 1 - delta ** (1 / n)

    b = (b1 + b2) / 2  # middle of segment
    if abs(binom_sum(b, e, n) - delta) < eps:
        return b
    elif binom_sum(b, e, n) <= delta - eps:
        return B_star(delta, e, n, b1=b1, b2=b)
    else:
        return B_star(delta, e, n, b1=b, b2=b2)


def integers_log_spacing(start, end, num_points=40):
    """
    Returns a list of integers between `start` and `end`, spaced so that more points
    are concentrated toward the start of the range (log-distributed shape).

    Parameters:
    - start (int): Start of the range.
    - end (int): End of the range.
    - num_points (int): Number of points to sample (default is 40).

    Returns:
    - list[int]: Integers biased toward the beginning of the range.
    """
    if start >= end:
        raise ValueError("Start must be less than end.")
    # Generate a range from 0 (dense) to 1 (sparse)
    lin = np.linspace(0, 1, num_points)
    # Apply inverse exponential shape (log-bias toward the start)
    log_bias = 1 - (1 - lin) ** 4  # This compresses more values at the start
    # Scale to range
    values = start + (end - start) * log_bias
    values = np.round(values).astype(int)
    # Remove duplicates
    values = np.unique(np.clip(values, start, end))

    return values.tolist()


def integers_exp_spacing(start, end, num_points=40):
    """
    Returns a list of integers between `start` and `end`, spaced so that more points
    are concentrated toward the high end of the range.

    Parameters:
    - start (int): Start of the range.
    - end (int): End of the range.
    - num_points (int): Number of points to sample (default is 40).

    Returns:
    - list[int]: Integers biased toward the end of the range.
    """
    if start >= end:
        raise ValueError("Start must be less than end.")
    # Generate a range of indices from 0 to 1
    lin = np.linspace(0, 1, num_points)
    # Exponential bias toward 1 (end of the range)
    exp_bias = lin**4  # Tune the exponent for more/less bias
    # Scale to actual range
    values = start + (end - start) * exp_bias
    values = np.round(values).astype(int)
    # Remove duplicates
    values = np.unique(np.clip(values, start, end))

    return values.tolist()


def simulate_sgp_dataset(n, high_conf_propor=0.7, seed=42):
    """
    Simulate a dataset with binary predictions (`y_true`, `y_pred`) and confidence scores (`kappa`).
    The probability of a mistake (`y_true != y_pred`) decreases as kappa increases.

    Parameters:
    -----------
    n : int
        Number of samples to generate.
    high_conf_propor : proportion of predictions with high confidence distribution
    seed : for reproducibility

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns:
        - 'y_true': True binary labels.
        - 'y_pred': Predicted labels (0 or 1).
        - 'kappa': Confidence score (Beta-distributed).
    """
    if seed is not None:
        np.random.seed(seed)

    # y_true: binary, balanced classes
    y_true = np.random.choice([0, 1], size=n)
    # Generate two confidence ditributions using Beta distribution
    kappa = np.empty(n)
    match = (
        np.random.rand(n) < high_conf_propor
    )  # draw high_conf_propor % of samples with high confidence predictions
    kappa[match] = beta.rvs(
        9, 1, size=match.sum()
    )  # High confidence beta distribution, mean=0.9, variance=8.2e-3
    kappa[~match] = beta.rvs(
        3, 2, size=(~match).sum()
    )  # Lower confidence distribution, mean=0.6, variance=0.02

    # accuracy = 0.7*0.9+0.3*0.6=0.81 in this setting

    # Create y_pred based on mistake probabilities
    y_pred = np.zeros(n)
    for i in range(n):
        if np.random.rand() < kappa[i]:  # very likely if kappa confidence is high
            y_pred[i] = y_true[i]  # correct prediction
        else:
            y_pred[i] = 1 - y_true[i]  # incorrect prediction

    # Create DataFrame
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "kappa": kappa})

    return df
