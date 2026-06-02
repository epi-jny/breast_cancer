import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import pickle
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
import math
import scipy.special
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
from abstention_module.math_utils import *
from abstention_module.preprocessing import *

# Parameters k1 and k2 from the paper (see Algo 1-2 resp.)
K2 = 50
K1 = int(np.log(K2 + 1) / np.log(2)) + 1
DELTA = 5e-3


def emp_errs_count(samples, loss="standard"):
    """Count empirical errors in `samples` for the given loss ('standard', 'FP', 'FN').

    Args:
    
        samples (pd.DataFrame): Must contain boolean/int columns `y_pred`, `y_true`.
        loss (str): Error type to count.

    Returns:
        int: Number of errors.
    """
    if loss == "standard":
        return (samples.y_pred != samples.y_true).sum()
    elif loss == "FP":
        return ((samples.y_pred == 1) & (samples.y_true == 0)).sum()
    elif loss == "FN":
        return ((samples.y_pred == 0) & (samples.y_true == 1)).sum()
    else:
        raise ValueError("metric must be either 'standard', 'FP' or 'FN'")


def emp_metric(samples, metric="standard"):
    """Compute an empirical classification metric on `samples`.

    Supports: 'standard', 'FP', 'FN', 'FPR', 'FNR', 'PPV', 'SE', 'SP'.

    Args:
        samples (pd.DataFrame): Must contain `y_pred`, `y_true`.
        metric (str): Metric name.

    Returns:
        float: Metric value.

    Raises:
        ValueError: If dataset is empty or metric is unknown.
    """
    if samples.shape[0] == 0:
        raise ValueError("no sample in dataset")
    if metric == "standard":
        return emp_errs_count(samples) / samples.shape[0]
    elif metric == "FP":
        return emp_errs_count(samples, loss="FP") / samples.shape[0]
    elif metric == "FN":
        return emp_errs_count(samples, loss="FN") / samples.shape[0]
    elif metric == "FPR":
        return emp_errs_count(samples, loss="FP") / (1 - samples.y_true).sum()
    elif metric == "FNR":
        return emp_errs_count(samples, loss="FN") / samples.y_true.sum()
    elif metric == "PPV":
        return (samples.y_pred * samples.y_true).sum() / samples.y_pred.sum()
    elif metric == "SE":
        return (samples.y_pred * samples.y_true).sum() / samples.y_true.sum()
    elif metric == "SP":
        return ((1 - samples.y_pred) * (1 - samples.y_true)).sum() / (
            1 - samples.y_true
        ).sum()
    else:
        raise ValueError(
            "metric must be in 'standard', 'FP','FN','FPR','FNR','PPV','SE','SP'"
        )


def upper_bound_denominator(metric, selected_samples, delta, n):
    """Denominator term for upper bounds of ratio metrics.

    Applies to: FPR, FNR, PPV, SE, SP.

    Args:
        metric (str): Metric name.
        selected_samples (pd.DataFrame): Selected subset with `y_pred`, `y_true`.
        delta (float): Confidence level.
        n (int): Total sample size.

    Returns:
        float: Denominator value.
    """
    d2 = np.sqrt(n * np.log(2 / delta) / 2) / selected_samples.shape[0]
    if metric == "PPV":
        d1 = selected_samples.y_pred.sum() / selected_samples.shape[0]
    else:
        d1 = selected_samples.y_true.sum() / selected_samples.shape[0]

    if metric in ["FPR", "SP"]:
        return 1 - d1 - d2
    else:  # FNR, SE, PPV
        return d1 - d2


def bound(b, selected_samples, delta, metric, n):
    """Transform risk bound `b` into a metric-specific bound.

    Args:
        b (float): Base bound B*.
        selected_samples (pd.DataFrame): Selected subset.
        delta (float): Confidence level.
        metric (str): Target metric.
        n (int): Total sample size.

    Returns:
        float: Metric bound
    """
    if metric in ["standard", "FP", "FN"]:
        B = b
    elif metric in ["FPR", "FNR"]:
        B = b / upper_bound_denominator(metric, selected_samples, delta, n)
    else:  # PPV, SE, SP
        B = 1 - b / upper_bound_denominator(metric, selected_samples, delta, n)

    return B


def satisfied(bound, r_star, metric):
    """Check if the target constraint is satisfied for the metric.

    Args:
        bound (float): Current bound.
        r_star (float): Target level.
        metric (str): Metric name.

    Returns:
        bool: True if constraint is met.
    """
    if metric in ["standard", "FP", "FN", "FPR", "FNR"]:
        return True if bound <= r_star else False
    else:
        return True if bound >= r_star else False


def sgp_dicho(delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k1=K1):
    """Dichotomy search for θ achieving an SGP bound near target r*.

    Args:
        delta (float): Confidence level.
        r_star (float): Target metric level.
        Sn (pd.DataFrame): Training set with `kappa`, `y_pred`, `y_true`.
        metric (str): Metric name.

    Returns:
        dict: {'theta_star','bound','delta','coverage','emp_metric'} or {} if none.
    """
    n = Sn.shape[0]

    for _ in range(k1):

        theta = (theta_min + theta_max) / 2
        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(selected_samples, loss=metric)
        b = B_star(delta / (2**k1 - 1), selected_errs_count, selected_samples.shape[0])
        if (selected_samples.shape[0] == 0) or (
            selected_errs_count == selected_samples.shape[0]
        ):
            b = 1  # by definition of B^*(.) in Proposition 1.

        if (b == 1) or (
            (selected_errs_count == 0) and (b >= r_star)
        ):  # terminal condition of Algo 1
            return {}
        else:
            if b < r_star:
                theta_max = theta
            else:
                theta_min = theta

    return {
        "theta_star": theta,
        "bound": b,
        "delta": delta,
        "coverage": selected_samples.shape[0] / n,
        "emp_metric": emp_metric(selected_samples, metric=metric),
    }


def sgp_greedy_search(delta, r_star, Sn, metric, theta_min=0.5, theta_max=1, k2=K2):
    """Greedy scan over θ to find the lowest θ satisfying the target bound.

    Args:
        delta (float): Confidence level.
        r_star (float): Target metric level.
        Sn (pd.DataFrame): Training set with `kappa`, `y_pred`, `y_true`.
        metric (str): Metric name.
        k2 (int): Grid size (Sn-independent).

    Returns:
        dict: {'theta_star','bound','delta','coverage','emp_metric'} or {} if none.
    """
    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }
    thetas = np.linspace(theta_min, theta_max, k2)[:-1]

    for theta in thetas:
        try:
            if selected_samples.shape[0] == 0:
                return {}
        except:
            pass

        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(
            selected_samples, loss=metric_loss_mapping[metric]
        )

        if selected_errs_count == 0:
            # no mistake on selected subset => no mistake as next iters, so b* is stuck at 1-delta^(1/n)
            return {}
        b = (
            B_star(delta / k2, selected_errs_count, selected_samples.shape[0])
            if (metric in ["standard", "FP", "FN"])
            else B_star(  # see formula in Prop 3
                delta / (2 * k2), selected_errs_count, selected_samples.shape[0]
            )
        )

        if (selected_samples.shape[0] == 0) or (
            selected_errs_count == selected_samples.shape[0]
        ):
            b = 1  # by definition of B^*(.) in Proposition 1.
        if b == 1:
            return {}

        B = bound(b, selected_samples, delta / k2, metric, n=Sn.shape[0])

        if satisfied(B, r_star, metric):
            return {
                "theta_star": theta,
                "bound": B,
                "delta": delta,
                "coverage": selected_samples.shape[0] / Sn.shape[0],
                "emp_metric": emp_metric(selected_samples, metric=metric),
            }

    return {}  # if we never found satisfactory B..


def sgp_at_targets(
    train_set,
    test_set,
    delta=DELTA,
    metric_targets=[i / 100 for i in range(1, 15)],
    metric="standard",
    mode="greedy",
    k2=K2,
    theta_min=0.5,
    theta_max=1,
):
    """Run SGP across multiple target levels and report train/test outcomes.

    Args:
        train_set (pd.DataFrame): Training data with `kappa`, `y_pred`, `y_true`.
        test_set (pd.DataFrame): Test data with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        metric_targets (list[float]): Target levels r*.
        metric (str): Metric name.
        mode (str): 'greedy' or 'dicho'.
        k2 (int): Grid size.

    Returns:
        pd.DataFrame: One row per target with bounds, θ*, and coverages.
    """
    results = []
    for r_star in metric_targets:

        if mode == "dicho":
            sgp_dico = sgp_dicho(
                delta,
                r_star,
                train_set,
                metric=metric,
                theta_min=theta_min,
                theta_max=theta_max,
            )
        elif mode == "greedy":
            sgp_dico = sgp_greedy_search(
                delta,
                r_star,
                train_set,
                metric,
                theta_min=theta_min,
                theta_max=theta_max,
                k2=k2,
            )
        else:
            raise ValueError('mode should be either "greedy" or "dicho"')

        if (
            sgp_dico != {} and abs(sgp_dico["bound"] - r_star) < 0.1
        ):  # we don't want the bound if it's too off target
            theta_star = sgp_dico["theta_star"]
            covered_test_set = test_set.loc[test_set.kappa > theta_star]
            if covered_test_set.shape[0] > 0:
                test_metric = emp_metric(covered_test_set, metric=metric)
            else:
                test_metric = np.nan
            results.append(
                {
                    "metric_target": r_star,
                    "metric_bound": sgp_dico["bound"],
                    "theta_star": theta_star,
                    "train_metric": sgp_dico["emp_metric"],
                    "train_coverage": sgp_dico["coverage"],
                    "test_metric": test_metric,
                    "test_coverage": covered_test_set.shape[0] / test_set.shape[0],
                }
            )

    return pd.DataFrame(results)


def sgp_at_targets_on_imbalanced_sets(
    proportions_of_1,
    metric_targets,
    sgp_df,
    delta=DELTA,
    mode="dicho",
    k2=K2,
    metric="standard",
):
    """Evaluate SGP at multiple class-1 proportions.

    Args:
        proportions_of_1 (list[float]): Desired positive-class rates.
        metric_targets (list[float]): Target levels r*.
        sgp_df (pd.DataFrame): Base dataset with `y_true`, `kappa`.
        delta (float): Confidence level.
        mode (str): 'greedy' or 'dicho'.
        k2 (int): Grid size.
        metric (str): Metric name.

    Returns:
        pd.DataFrame: Results with proportion, bounds, θ*, and metrics.
    """
    all_propor_dfs = pd.DataFrame()
    imbalanced_datasets = generate_imbalanced_datasets(sgp_df, proportions_of_1, seed=0)

    for proportion_1, imbalanced_set in zip(proportions_of_1, imbalanced_datasets):

        train_set_ = imbalanced_set.iloc[: int(imbalanced_set.shape[0] / 2)]
        train_set_ = (
            train_set_.sort_values("kappa", ascending=True)
            .reset_index(drop=True)
            .copy()
        )
        test_set_ = imbalanced_set.iloc[int(imbalanced_set.shape[0] / 2) :]

        results = sgp_at_targets(
            train_set_,
            test_set_,
            delta=delta,
            metric_targets=metric_targets,
            metric=metric,
            mode=mode,
            k2=k2,
        )
        results["proportion_1"] = proportion_1
        all_propor_dfs = pd.concat([all_propor_dfs, results]).reset_index(drop=True)

    return all_propor_dfs


def bound_evo_w_theta(
    metric, Sn, delta, theta_min=0.5, theta_max=1, k2=K2, frac_details=False
):
    """Trace the metric bound as a function of θ.

    Args:
        metric (str): Metric name.
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        k2 (int): Grid size.

    Returns:
        (np.ndarray, list[float]): (thetas, bounds) with NaNs for invalid regions.
    """
    metric_loss_mapping = {
        "standard": "standard",
        "FP": "FP",
        "FN": "FN",
        "FPR": "FP",
        "FNR": "FN",
        "PPV": "FP",
        "SE": "FN",
        "SP": "FP",
    }
    Sn = Sn.sort_values("kappa", ascending=True)
    bounds, thetas = [], np.linspace(theta_min, theta_max, k2)
    numerators, denominators = [], []

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        selected_errs_count = emp_errs_count(
            selected_samples, loss=metric_loss_mapping[metric]
        )
        if selected_errs_count == 0:
            break

        b = (
            B_star(delta / k2, selected_errs_count, selected_samples.shape[0])
            if (metric in ["standard", "FP", "FN"])
            else B_star(
                delta / (2 * k2), selected_errs_count, selected_samples.shape[0]
            )
        )

        if (selected_samples.shape[0] == 0) or (
            selected_errs_count == selected_samples.shape[0]
        ):
            b = 1  # by definition of B^*(.) in Proposition 1.
        if b == 1:
            break

        B = bound(b, selected_samples, delta / k2, metric, n=Sn.shape[0])
        d = upper_bound_denominator(metric, selected_samples, delta / k2, Sn.shape[0])
        if d < 0:  # => bound is negative or greater than 1 (non-informative)
            break
        if frac_details:
            numerators.append(b)
            denominators.append(d)

        bounds.append(B)

    bounds = bounds[:-1]  # because at theta max selected set is empty
    while len(bounds) < len(thetas):
        bounds.append(np.nan)
        if frac_details:
            numerators.append(np.nan)
            denominators.append(np.nan)

    if frac_details:
        return thetas, bounds, numerators, denominators
    return thetas, bounds


def reachable_bounds(metrics_list, Sn, delta=DELTA, theta_min=0.5, theta_max=1, k2=K2):
    """Compute θ/coverage grids and bounds for a list of metrics.

    Args:
        metrics_list (list[str]): Metrics to evaluate.
        Sn (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        k2 (int): grid size.

    Returns:
        dict: {'thetas','coverages', metric->bounds}.
    """
    res_dico = {}

    # thetas and coverages coordinates
    thetas = np.linspace(theta_min, theta_max, k2)
    res_dico["thetas"] = sorted(thetas)
    res_dico["coverages"] = sorted(
        [Sn.loc[Sn.kappa >= theta].shape[0] / Sn.shape[0] for theta in thetas],
        reverse=True,
    )
    # metrics bounds with respect to thetas
    for metric in metrics_list:
        _, bounds = bound_evo_w_theta(
            metric, Sn, delta, theta_min=theta_min, theta_max=theta_max, k2=k2
        )
        res_dico[metric] = bounds

    return res_dico


def pos_propor_w_theta(Sn, k2=K2, theta_min=0.5, theta_max=1):
    """Compute positive-class proportion among samples selected by θ.

    Args:
        Sn (pd.DataFrame): Dataset with `kappa`, `y_true`.
        k2 (int): grid size.

    Returns:
        (np.ndarray, list[float]): (thetas, positive proportions).
    """
    Sn = Sn.sort_values("kappa", ascending=True)
    pos_propor, thetas = [], np.linspace(theta_min, theta_max, k2)

    for theta in thetas:

        selected_samples = Sn.loc[Sn.kappa >= theta]
        pos_propor.append(selected_samples.y_true.sum() / selected_samples.shape[0])

    return thetas, pos_propor


def runtime(sim_df, mode: str = "dicho", k2: int = K2, theta_min=0.5, theta_max=1):
    """Measure wall-time (seconds) for SGP search mode on `sim_df`.

    Args:
        sim_df (pd.DataFrame): Simulated Dataset for timing.
        mode (str): 'dicho' or 'greedy'.
        k2 (int): grid size.

    Returns:
        int: Elapsed seconds.
    """
    t0 = datetime.now()
    if mode == "dicho":
        res = sgp_dicho(
            delta=DELTA,
            r_star=0.05,
            Sn=sim_df,
            metric="standard",
            theta_min=theta_min,
            theta_max=theta_max,
        )
    elif mode == "greedy":
        res = sgp_greedy_search(
            delta=DELTA,
            r_star=0.05,
            Sn=sim_df,
            metric="standard",
            theta_min=theta_min,
            theta_max=theta_max,
            k2=k2,
        )
    else:
        raise ValueError("mode should either be dicho or greedy")
    t1 = datetime.now()
    return (t1 - t0).seconds


def joint_control(
    metrics_and_targets,
    sgp_df,
    delta=DELTA,
    theta_min=0.5,
    theta_max=1,
    plot=False,
    k2=K2,
):
    """Find θ intervals satisfying multiple metric targets (optionally plot).

    Args:
        metrics_and_targets (dict): {metric: target}.
        sgp_df (pd.DataFrame): Dataset with `kappa`, `y_pred`, `y_true`.
        delta (float): Confidence level.
        plot (bool): If True, plot bounds and feasible θ segments.
        k2 (int): grid size.

    Returns:
        dict | None: If not plotting, {'theta_intervals', 'best_theta'}.
    """
    metric_sign_mapping = {
        "standard": "<",
        "FP": "<",
        "FN": "<",
        "FPR": "<",
        "FNR": "<",
        "PPV": ">",
        "SE": ">",
        "SP": ">",
    }
    y_proj = -0.01
    projection_handles = []

    # Use a colormap to assign a unique color to each metric
    num_metrics = len(metrics_and_targets)
    color_map = cm.get_cmap("tab10", num_metrics)
    colors = [color_map(i) for i in range(num_metrics)]
    segments_per_metric = {key: [] for key in metrics_and_targets.keys()}

    if plot:
        plt.figure()

    for i, (metric, target) in enumerate(metrics_and_targets.items()):
        thetas, bounds = bound_evo_w_theta(
            metric, sgp_df, delta, theta_min=theta_min, theta_max=theta_max, k2=k2
        )
        color = colors[i]

        if metric_sign_mapping[metric] == ">":
            mask = np.array(bounds) > target
        else:
            mask = np.array(bounds) < target

        segments = get_segments(thetas, mask)
        segments_per_metric[metric] = segments

        if plot:
            plt.plot(
                thetas, bounds, color=color, label=f"{metric} bound", linewidth=1.5
            )
            plt.axhline(y=target, color=color, linestyle="--", label=f"{metric} target")
            plt.xlabel(r"$\theta$")
            plt.ylabel("Metric bounds")
            plt.tick_params(axis="y")

            for x_start, x_end in segments:
                plt.hlines(y_proj * i, x_start, x_end, colors=color, linewidth=2)

            # Legend handle for projections
            projection_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2,
                    label=r"$\theta$ "
                    + f"/ {metric} {metric_sign_mapping[metric]} {target}",
                )
            )

    if plot:
        plt.ylim(bottom=y_proj - 0.02)
        plt.legend(handles=projection_handles, loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        intersected_intervals = compute_all_interval_intersections(segments_per_metric)
        return {
            "theta_intervals": intersected_intervals,
            "best_theta": best_theta(intersected_intervals),
        }


def mean_abs_diff(u, v):
    """Mean absolute difference ignoring NaNs (pairwise).

    Args:
        u, v (array-like): Sequences to compare.

    Returns:
        float: Mean |u - v| over valid pairs, or NaN if none.
    """
    u = np.asarray(u)
    v = np.asarray(v)

    # Mask to filter out nan entries
    not_masked = ~np.isnan(u) & ~np.isnan(v)

    # If no valid entries, return nan
    if not np.any(not_masked):
        return np.nan

    diffs = np.abs(u[not_masked] - v[not_masked])
    return np.mean(diffs)


def ABC(ds, metric, theta_min=0.5, theta_max=1, k2=30, delta=DELTA):
    """Compute average absolute gap between bound and test metric vs θ.

    Args:
        ds (pd.DataFrame): Dataset split in half into train/test.
        metric (str): One of {'standard','FP','FN','FPR','FNR'}.
        k2 (int): grid size.
        delta (float): Confidence level.

    Returns:
        float: Mean absolute difference between bound and empirical metric.
    """
    train_set = ds.iloc[: int(len(ds) / 2)]
    train_set = (
        train_set.sort_values("kappa", ascending=True).reset_index(drop=True).copy()
    )
    test_set = ds.iloc[int(len(ds) / 2) :]

    thetas, bounds = bound_evo_w_theta(
        metric, train_set, delta, theta_min=theta_min, theta_max=theta_max, k2=k2
    )
    emp_metrics = []
    for theta in thetas:
        try:
            selected_set = test_set.loc[test_set.kappa >= theta].copy()
            emp_metrics.append(emp_metric(selected_set, metric=metric))
        except ValueError:
            emp_metrics.append(np.nan)

    return mean_abs_diff(bounds, emp_metrics)


def our_bound(selected_samples, metric, n, delta=DELTA):
    """
    Compute our guaranteed conditional metric bound (to be compared to external reference)

    Args:
        selected_samples: samples with confidence higher than threshold
        metric: one of the selective metrics 'standard', 'FPR', 'FNR' etc...
        delta: probability control
        n: size of Sn, the original dataset

    Returns:
        float: bound from proposition 2-3
    """
    loss = "FP" if (metric in ["FPR", "PPV"]) else "FN"
    selected_errs_count = emp_errs_count(selected_samples, loss=loss)
    b = B_star(delta / 2, selected_errs_count, selected_samples.shape[0])
    B = bound(b, selected_samples, delta, metric, n=n)

    return B if (B > 0 and B < 1) else np.nan


def eq11_bound(selected_samples, metric, delta=DELTA):
    """
    Compute conditional metric bound with Eq. (11) from (Balsubramani et al., 2019)

    Args:
        selected_samples: samples with confidence higher than threshold
        metric: 'FPR', 'FNR' etc..
        delta: probability control

    Returns:
        float: bound from Eq. (11)
    """
    if metric == "FPR":
        a = (selected_samples.y_pred * (1 - selected_samples.y_true)).sum() / (
            1 - selected_samples.y_true
        ).sum()
        b = np.sqrt(-2 * np.log(delta) / (1 - selected_samples.y_true).sum())

    elif metric == "FNR":
        a = (
            (1 - selected_samples.y_pred) * selected_samples.y_true
        ).sum() / selected_samples.y_true.sum()
        b = np.sqrt(-2 * np.log(delta) / selected_samples.y_true.sum())

    else:  # PPV
        a = (
            selected_samples.y_pred * selected_samples.y_true
        ).sum() / selected_samples.y_pred.sum()
        b = np.sqrt(-2 * np.log(delta) / selected_samples.y_pred.sum())
        return a - b

    return a + b
