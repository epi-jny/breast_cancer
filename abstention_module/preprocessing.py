import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
import math
import scipy.special
import random as rd
from itertools import product
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
from datetime import datetime
from sklearn.utils import resample


def filter_classes(dataset, allowed_classes):
    """
    Function to filter CIFAR dataset by class labels
    """
    indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_classes]
    return Subset(dataset, indices)


def binarize_labels(dataset):
    """
    Binarize labels: 1 if airplane (class index 0), else 0
    """
    dataset.targets = [1 if label == 0 else 0 for label in dataset.targets]


def get_subset_labels(subset, combined_dataset):
    """
    Extract labels for the subsets (needed for sampling)
    """
    return [combined_dataset[idx][1] for idx in subset.indices]


def get_balanced_sampler(labels):
    """
    Compute weights for balanced sampling
    """
    class_sample_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_sample_counts.float()
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def prepare_sgp_dico(dataloader, model, device, T):
    """
    Prepare dataframe containing exactly the required features to train sgp module
    true class of each sample, model predicted class,
    softmax response (kappa) or any other confidence function output
    """
    sgp_dico = {"y_true": np.array([]), "y_pred": np.array([]), "kappa": np.array([])}
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            batch_preds = model(images)
            softmax_values = F.softmax(batch_preds / T, dim=1)
            softmax_responses = torch.max(softmax_values, dim=1)[0].cpu().numpy()
            _, predicted_classes = torch.max(batch_preds, 1)
            predicted_classes = predicted_classes.cpu().numpy()

            sgp_dico["y_true"] = np.concatenate(
                (sgp_dico["y_true"], labels.cpu().numpy())
            )
            sgp_dico["y_pred"] = np.concatenate((sgp_dico["y_pred"], predicted_classes))
            sgp_dico["kappa"] = np.concatenate((sgp_dico["kappa"], softmax_responses))

    return sgp_dico


def generate_imbalanced_datasets(
    dataset, proportions, label_col="y_true", seed=0, fixed=False
):
    """
    Create datasets with specified class-1 proportions.

    When fixed=False (default):
        - Reproduces the original adaptive logic: tries to achieve the target proportion by
          downsampling the majority class; if not possible, downsample the minority instead.
        - Resulting dataset sizes may differ from the original.

    When fixed=True:
        - Always returns datasets with the same total size as the input.
        - Achieves the target proportion by downsampling the majority class when possible and
          oversampling (with replacement) the minority class when necessary.

    Args:
        dataset (pd.DataFrame): Input DataFrame with a binary label column.
        proportions (list[float|None]): Desired class-1 proportions; `None` returns a shuffled copy.
        label_col (str): Name of the label column.
        seed (int): Random seed for reproducibility.
        fixed (bool): If True, keep the generated dataset size equal to len(dataset).

    Returns:
        list[pd.DataFrame]: One dataset per requested proportion.
    """
    # Use a dedicated RNG so multiple .sample() calls advance deterministically.
    rng = np.random.RandomState(seed)

    df_pos = dataset[dataset[label_col] == 1]
    df_neg = dataset[dataset[label_col] == 0]

    N1 = len(df_pos)
    N0 = len(df_neg)
    N_orig = len(dataset)

    datasets = []

    for p in proportions:
        if p is None:
            # Just return a shuffled copy of the original (still fixed-size).
            datasets.append(
                dataset.sample(frac=1, random_state=rng).reset_index(drop=True)
            )
            continue

        if not (0 < p < 1):
            raise ValueError(f"Proportion must be between 0 and 1 (exclusive). Got {p}")

        if fixed:
            # Target counts constrained to original total size.
            # Use round, but clamp to [1, N_orig-1] to avoid degenerate all-one-class results.
            N1_target = int(round(p * N_orig))
            N1_target = max(1, min(N_orig - 1, N1_target))
            N0_target = N_orig - N1_target

            # Sample positives
            if N1_target <= N1:
                df_pos_sampled = df_pos.sample(
                    n=N1_target, replace=False, random_state=rng
                )
            else:
                # Oversample with replacement to reach target
                df_pos_sampled = df_pos.sample(
                    n=N1_target, replace=True, random_state=rng
                )

            # Sample negatives
            if N0_target <= N0:
                df_neg_sampled = df_neg.sample(
                    n=N0_target, replace=False, random_state=rng
                )
            else:
                # Oversample with replacement to reach target
                df_neg_sampled = df_neg.sample(
                    n=N0_target, replace=True, random_state=rng
                )

            df_combined = pd.concat([df_pos_sampled, df_neg_sampled], axis=0)

        else:
            # Original adaptive downsampling behavior (sizes may differ from N_orig).
            if N1 == 0 or N0 == 0:
                raise ValueError(
                    "Both classes must be present in the input dataset for adaptive mode."
                )

            # Target total size to achieve proportion p of class-1 by downsampling majority.
            N_total = int(N1 / p)
            N0_required = N_total - N1

            if N0_required <= N0:
                # Downsample class-0
                df_neg_sampled = df_neg.sample(
                    n=N0_required, replace=False, random_state=rng
                )
                df_combined = pd.concat([df_pos, df_neg_sampled], axis=0)
            else:
                # Not enough class-0s — fallback to downsampling class-1 based on full dataset size
                N1_required = int(p * (N0 + N1))
                df_pos_sampled = df_pos.sample(
                    n=min(N1_required, N1), replace=False, random_state=rng
                )
                df_combined = pd.concat([df_pos_sampled, df_neg], axis=0)

        # Shuffle and reset index
        df_shuffled = df_combined.sample(frac=1, random_state=rng).reset_index(
            drop=True
        )
        datasets.append(df_shuffled)

    return datasets


def compute_mean_std(dataset_root):
    """
    Compute the mean and standard deviation of an image dataset for normalization purposes.

    Args:
        dataset_root (str): Root directory path of the image dataset (organized in class-specific subfolders).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors for the dataset (per channel).
    """
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )

    dataset = ImageFolder(root=dataset_root, transform=transform)
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def split_and_balance_dataset(
    root, transform=None, seed=0, train_size=0.4, val_size=0.1
):
    """
    Split a binary image classification dataset into train (40%), val (10%), and test (50%) subsets,
    then oversample minority class in train and val sets for balance.

    Args:
        root (str): Root directory of the dataset (subfolders for each class).
        transform (callable, optional): Transformations to apply to images.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Subset, Subset, Subset]: Balanced train, balanced val, and original test subsets.
    """
    dataset = ImageFolder(root=root, transform=transform)
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)

    # Ensure binary classification
    classes = np.unique(targets)
    assert len(classes) == 2, "Expected binary classification dataset (2 classes)."

    # Split indices into train (40%), val (10%), test (50%)
    indices = np.arange(len(targets))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Helper to oversample a binary split
    def oversample(indices_subset):
        labels = targets[indices_subset]
        class0_indices = [i for i in indices_subset if targets[i] == 0]
        class1_indices = [i for i in indices_subset if targets[i] == 1]

        # Determine majority and minority
        if len(class0_indices) > len(class1_indices):
            majority, minority = class0_indices, class1_indices
        else:
            majority, minority = class1_indices, class0_indices

        # Oversample minority to match majority
        oversampled_minority = resample(
            minority, replace=True, n_samples=len(majority), random_state=seed
        )
        balanced_indices = majority + oversampled_minority
        rng.shuffle(balanced_indices)
        return balanced_indices

    # Balance train and val sets
    train_indices_balanced = oversample(train_indices)
    val_indices_balanced = oversample(val_indices)

    # Return as PyTorch Subsets
    return (
        Subset(dataset, train_indices_balanced),
        Subset(dataset, val_indices_balanced),
        Subset(dataset, test_indices),
    )


def count_labels(dataset):
    """
    Count the number of samples for each label (0 and 1) in a dataset.

    Args:
        dataset (Dataset or Subset): A PyTorch Dataset or Subset (e.g., from ImageFolder or a split subset).

    Returns:
        dict: A dictionary mapping class labels (0 and 1) to their respective counts.
    """
    targets = []

    # Handle Subset datasets
    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    counts = Counter(targets)
    return {label: counts.get(label, 0) for label in range(2)}


def interval_intersection(intervals):
    """
    Given a list of intervals [(start1, end1), (start2, end2), ...],
    returns their intersection as (start_max, end_min), or None if empty.
    """
    start = max(interval[0] for interval in intervals)
    end = min(interval[1] for interval in intervals)
    if start < end:
        return (start, end)
    return None  # No overlap


def compute_all_interval_intersections(interval_dict):
    """
    Given a dict of {key: [(start, end), ...]}, computes all non-empty intersections
    where one interval is selected from each key.
    """
    # Generate all combinations: one interval from each key
    keys = list(interval_dict.keys())
    interval_lists = [interval_dict[key] for key in keys]

    intersections = []
    for combo in product(*interval_lists):
        inter = interval_intersection(combo)
        if inter is not None:
            intersections.append(inter)

    return intersections


def best_theta(intersection_intervals):
    """
    Return the smallest theta value from a list of intervals.

    Args:
        intersection_intervals (list of tuple): Intervals of the form (theta, ...).

    Returns:
        float: The minimum theta found, or np.inf if input is invalid/empty.
    """
    l = 0
    m = np.inf
    try:
        while l < len(intersection_intervals):
            if (
                intersection_intervals[l][0] < m
            ):  # looking for smallest theta through all the intervals
                m = intersection_intervals[l][0]
            l += 1
    except TypeError:
        pass
    return m


def get_segments(x, condition_mask):
    """
    Identify and return contiguous segments in `x` where `condition_mask` is True.

    Parameters:
        x (list): A list of values (e.g., time or index values).
        condition_mask (list of bool): A boolean list of the same length as `x`,
                                       indicating which elements satisfy the condition.

    Returns:
        list of tuples: A list of (start, end) pairs representing segments where
                        `condition_mask` is True.
    """
    segments = []
    in_segment = False
    for i in range(len(x)):
        if condition_mask[i] and not in_segment:
            start = x[i]
            in_segment = True
        elif not condition_mask[i] and in_segment:
            end = x[i - 1]
            segments.append((start, end))
            in_segment = False
    if in_segment:
        segments.append((start, x[-1]))
    return segments


def train_test_split(df, seed, p_train=0.75):
    """
    Standard dataset split function

    Args:
        df: dataset to split (pandas)
        seed: int for reproducible results
        p_train: proportion of df going to train set
    """
    train_df = df.sample(frac=p_train, random_state=seed)
    test_df = df.drop(train_df.index)
    train_df = train_df.sort_values("kappa", ascending=True).reset_index(drop=True)
    return train_df, test_df
