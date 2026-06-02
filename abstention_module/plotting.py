import os
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math
import scipy.special
import random as rd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from matplotlib.lines import Line2D
from matplotlib import MatplotlibDeprecationWarning
from abstention_module.sgp_utils import *
from matplotlib.ticker import AutoMinorLocator
from abstention_module.sgp_utils import DELTA

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def metric_plots(
    all_results: dict,
    metric: str,
    lines_list: list,
    xlim: list = [0, 1],
    ylim: list = [0, 1],
    title: str = "",
    save_path: str = None,
    show: bool = True,
):
    """
    Plot one or more metric curves from experiment results.

    Args:
        save_path: si fourni, enregistre la figure à ce chemin (PNG/PDF/SVG selon ext).
        show: si False, ne fait pas `plt.show()` (utile en batch / serveur sans display).
    """

    plt.figure(figsize=(6, 4))  # Larger, balanced figure size

    for line in lines_list:
        kappa, x_axis, colname, name, c, style = (
            line["kappa"],
            line["x_axis"],
            line["colname"],
            line["name"],
            line["colour"],
            line["style"],
        )
        plt.plot(
            all_results[kappa][x_axis],
            all_results[kappa][colname],
            label=name,
            c=c,
            linestyle=style,
            linewidth=2,  # thicker lines
            alpha=0.9,  # slight transparency
            marker="o",
            markersize=3,
        )

    plt.xlabel(line["x_axis_name"], fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    plt.legend(
        frameon=True, facecolor="white", edgecolor="black", fontsize=10, loc="best"
    )

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)  # subtle dashed grid

    if len(title) > 0:
        plt.title(title, fontsize=14, weight="bold", loc="center")

    plt.tight_layout()
    if save_path is not None:
        from pathlib import Path as _P
        _P(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Figure écrite → {save_path}")
    if show:
        plt.show()
    plt.close()


def sgp_metric_block(
    all_results,
    metric_label,
    slug,
    run_dir,
    theta_SR,
    theta_MCD,
    direction="<",
    show=False,
):
    """Produit les figures metric_plots d'UNE métrique, façon notebook.

    Reproduit le bloc répété du notebook : pour une métrique donnée,
      - SR vs θ, SR vs coverage                       (lines_list_1_SR / _2_SR)
      - MCD vs θ, MCD vs coverage    (si "MCD" présent) (lines_list_1_MCD / _2_MCD)
      - SR v. MCD vs coverage        (si SR et MCD)      (lines_list_2)

    Args:
        all_results (dict): {"SR": df} ou {"SR": df, "MCD": df} (sorties sgp_at_targets).
        metric_label (str): libellé y (ex "0/1 risk", "FPR", "SE").
        slug (str): base du nom de fichier (ex "standard", "FPR").
        run_dir (str|Path): dossier du run où sauver les PNG.
        theta_SR, theta_MCD (tuple): (theta_min, theta_max) par fonction de confiance.
        direction (str): "<" (standard/FP/FN/FPR/FNR) ou ">" (PPV/SE/SP), pour les ylim.
        show (bool): afficher les figures (False en batch).

    Une métrique vide (sgp_at_targets sans résultat) est ignorée proprement.
    """
    run_dir = Path(run_dir)
    l1_SR, l2_SR, l1_MCD, l2_MCD, l2 = lines()

    def _ylim(res_list):
        tt = pd.concat([r.test_metric for r in res_list] + [r.train_metric for r in res_list])
        bd = pd.concat([r.metric_bound for r in res_list])
        tt = tt.replace([np.inf, -np.inf], np.nan).dropna()
        bd = bd.replace([np.inf, -np.inf], np.nan).dropna()
        if tt.empty or bd.empty:
            return None
        if direction == "<":
            return [0.99 * float(tt.min()), 1.01 * float(bd.max())]
        return [0.99 * float(bd.min()), min(1.0, 1.01 * float(tt.max()))]

    res_SR = all_results.get("SR")
    res_MCD = all_results.get("MCD")

    if res_SR is not None and not res_SR.empty:
        yl = _ylim([res_SR])
        metric_plots(
            all_results, metric=metric_label, lines_list=l1_SR,
            xlim=[0.99 * theta_SR[0], 1.01 * theta_SR[1]], ylim=yl,
            title=f"{metric_label} — SR", show=show,
            save_path=str(run_dir / f"sgp_{slug}_SR_vs_theta.png"),
        )
        metric_plots(
            all_results, metric=metric_label, lines_list=l2_SR,
            xlim=[0.99 * float(res_SR.test_coverage.min()),
                  1.01 * float(res_SR.test_coverage.max())], ylim=yl,
            title=f"{metric_label} — SR", show=show,
            save_path=str(run_dir / f"sgp_{slug}_SR_vs_coverage.png"),
        )

    if res_MCD is not None and not res_MCD.empty:
        yl = _ylim([res_MCD])
        metric_plots(
            all_results, metric=metric_label, lines_list=l1_MCD,
            xlim=[1.01 * theta_MCD[0], 1.01 * theta_MCD[1]], ylim=yl,
            title=f"{metric_label} — MCD", show=show,
            save_path=str(run_dir / f"sgp_{slug}_MCD_vs_theta.png"),
        )
        metric_plots(
            all_results, metric=metric_label, lines_list=l2_MCD,
            xlim=[0.99 * float(res_MCD.test_coverage.min()),
                  1.01 * float(res_MCD.test_coverage.max())], ylim=yl,
            title=f"{metric_label} — MCD", show=show,
            save_path=str(run_dir / f"sgp_{slug}_MCD_vs_coverage.png"),
        )

    if (res_SR is not None and not res_SR.empty
            and res_MCD is not None and not res_MCD.empty):
        yl = _ylim([res_SR, res_MCD])
        cov_min = min(float(res_SR.test_coverage.min()), float(res_MCD.test_coverage.min()))
        cov_max = max(float(res_SR.test_coverage.max()), float(res_MCD.test_coverage.max()))
        metric_plots(
            all_results, metric=metric_label, lines_list=l2,
            xlim=[0.99 * cov_min, 1.01 * cov_max], ylim=yl,
            title=f"{metric_label} — SR v. MCD", show=show,
            save_path=str(run_dir / f"sgp_{slug}_SR_vs_MCD_coverage.png"),
        )


def metric_plots_with_imbalance(
    all_propor_dfs,
    proportions,
    ylabel: str,
    ylim: list = [0, 1],
    xlim1: list = [0, 1],
    xlim2: list = [0, 1],
    title: str = None,
    show_left_legend=False,
):
    """
    Plot metric curves under varying class imbalance conditions.

    Produces two side-by-side plots:
      - Coverage vs metric.
      - θ* vs metric.
    Curves are drawn for each specified imbalance proportion, with color
    intensity indicating imbalance level.

    Args:
        all_propor_dfs (pd.DataFrame): Data with columns
            ['proportion_1', 'test_coverage', 'metric_bound',
             'test_metric', 'theta_star'].
        proportions (list): Proportion values of the positive class to plot.
        ylabel (str): Y-axis label (metric name).
        ylim (list, optional): Y-axis limits. Default [0, 1].
        xlim1 (list, optional): X-axis limits for coverage plot. Default [0, 1].
        xlim2 (list, optional): X-axis limits for θ* plot. Default [0, 1].
        title (str, optional): Title for the figure. Default None.
        show_left_legend (bool, optional): Whether to show legend and colorbar
            in the left subplot. Default False.

    Returns:
        None. Displays the plots.
    """

    # Set up colormaps
    cmap_blue = cm.get_cmap("Blues")
    cmap_orange = cm.get_cmap("Oranges")
    cmap_gray = cm.get_cmap("Grays")

    # Normalize for colorbar
    norm = mcolors.Normalize(vmin=1, vmax=50)
    sm = cm.ScalarMappable(cmap=cmap_gray, norm=norm)
    sm.set_array([])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Track for legend proxies
    proxy_blue = plt.Line2D(
        [0], [0], color=cmap_blue(0.8), label="Guaranteed", linestyle="--"
    )
    proxy_orange = plt.Line2D([0], [0], color=cmap_orange(0.8), label="On test set")

    for proportion_1 in proportions:
        norm_value = (10 + proportion_1 * 100) / 60
        color_blue = cmap_blue(norm_value)
        color_orange = cmap_orange(norm_value)

        results = all_propor_dfs.loc[all_propor_dfs.proportion_1 == proportion_1]

        # Coverage subplot
        axs[0].plot(
            results.test_coverage,
            results.metric_bound,
            color=color_blue,
            linestyle="--",
        )
        axs[0].plot(results.test_coverage, results.test_metric, color=color_orange)

        # Theta* subplot
        axs[1].plot(
            results.theta_star, results.metric_bound, color=color_blue, linestyle="--"
        )
        axs[1].plot(results.theta_star, results.test_metric, color=color_orange)

    # Labels and limits
    axs[0].set_xlabel("Coverage")
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlim(xlim1)
    axs[0].set_ylim(ylim)
    axs[0].grid(True)
    if show_left_legend:
        cbar = fig.colorbar(sm, ax=axs[0], shrink=0.95)
        cbar.set_label("Proportion of 1s (%)")
        axs[0].legend(handles=[proxy_blue, proxy_orange])

    axs[1].set_xlabel(r"$\theta^*$")
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlim(xlim2)
    axs[1].set_ylim(ylim)
    axs[1].grid(True)
    axs[1].legend(handles=[proxy_blue, proxy_orange])

    # Shared colorbar
    cbar = fig.colorbar(sm, ax=axs[1], shrink=0.95)
    cbar.set_label("Proportion of 1s (%)")

    plt.tight_layout()
    if title:
        plt.title(title, loc="center")
    plt.show()


def show_cifar10(t: torch.Tensor, title=None):
    """
    Display a CIFAR-10 tensor image (normalized with CIFAR-10 stats).
    t: shape (C,H,W) or (H,W,C), values normalized.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    x = t.detach().cpu().float()

    # ensure CHW
    if x.ndim == 3 and x.shape[0] in (1, 3):
        chw = x
    elif x.ndim == 3 and x.shape[2] in (1, 3):
        chw = x.permute(2, 0, 1)
    else:
        raise ValueError("Expected (C,H,W) or (H,W,C) with C=1 or 3.")

    # unnormalize CIFAR-10
    chw = chw * std + mean

    # clamp to [0,1]
    img = chw.clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_all_metrics(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    color_map: dict,
    title: str = "",
    delta: float = DELTA,
    xlim1: list = [0, 1],
    xlim2: list = [0, 1],
    ylim1: list = [0, 1],
    ylim2: list = [0, 1],
    by_coverage: bool = False,
    metrics: list = ["standard", "FP", "FN", "FPR", "FNR", "PPV", "SE", "SP"],
    theta_min=0.5,
    theta_max=1,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot training bounds and test metrics across thresholds or coverages.

    Args:
        save_path: si fourni, enregistre la figure à ce chemin (PNG/PDF/SVG selon ext).
        show: si False, ne fait pas `plt.show()` (utile en batch / serveur sans display).
    """

    rc = {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D0D0D0",
        "lines.linewidth": 2.0,
        "grid.alpha": 0.4,
    }

    with plt.rc_context(rc), plt.style.context("seaborn-v0_8-whitegrid"):

        label_map = {
            "standard": "0/1 risk",
            "FP": "FP risk",
            "FN": "FN risk",
            "FPR": "FPR",
            "FNR": "FNR",
        }

        def _beautify_ax(ax):
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#B0B0B0")
                spine.set_linewidth(0.9)
            ax.tick_params(which="both", length=4, width=0.8)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

        def plot_metrics_on_ax(ax, metrics_subset, xlim, ylim, label_getter):
            last_thetas = None
            for metric in metrics_subset:
                color = color_map[metric]
                thetas, bounds = bound_evo_w_theta(
                    metric,
                    train_set,
                    delta=DELTA,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    k2=50,
                )
                last_thetas = thetas

                # Training bound
                if by_coverage:
                    train_coverages = [
                        train_set.loc[train_set.kappa >= theta].shape[0]
                        / train_set.shape[0]
                        for theta in thetas
                    ]
                    idx = np.argsort(train_coverages)
                    x_train, y_train = (
                        np.array(train_coverages)[idx],
                        np.array(bounds)[idx],
                    )
                else:
                    x_train, y_train = thetas, bounds
                ax.plot(
                    x_train,
                    y_train,
                    color=color,
                    label=label_getter(metric) + " bound",
                    linewidth=2,
                    alpha=0.95,
                )

                # Test empirical metric
                emp_metrics, test_coverages = [], []
                for theta in thetas:
                    selected_set = test_set.loc[test_set.kappa >= theta].copy()
                    test_coverages.append(selected_set.shape[0] / test_set.shape[0])
                    try:
                        emp_metrics.append(emp_metric(selected_set, metric=metric))
                    except ValueError:
                        emp_metrics.append(np.nan)

                if by_coverage:
                    idx = np.argsort(test_coverages)
                    x_test, y_test = (
                        np.array(test_coverages)[idx],
                        np.array(emp_metrics)[idx],
                    )
                else:
                    x_test, y_test = thetas, emp_metrics
                ax.plot(
                    x_test,
                    y_test,
                    linestyle="--",
                    color=color,
                    label="Test " + label_getter(metric),
                    linewidth=2,
                    alpha=0.95,
                )

            if by_coverage:
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_xlabel("Coverage")
            else:
                ax.set_xlim(min(last_thetas), max(last_thetas))
                ax.set_xlabel(r"$\theta$")
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_ylabel("Metric value")

            # Clean legend
            ax.legend(
                loc="best", ncols=1, handlelength=2.5, borderpad=0.6, labelspacing=0.5
            )

            _beautify_ax(ax)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("white")

        left_group = [m for m in metrics if m in ["standard", "FP", "FN", "FPR", "FNR"]]
        right_group = [m for m in metrics if m in ["PPV", "SE", "SP"]]

        if left_group:
            plot_metrics_on_ax(
                axes[0], left_group, xlim1, ylim1, lambda m: label_map[m]
            )
        if right_group:
            plot_metrics_on_ax(axes[1], right_group, xlim2, ylim2, lambda m: m)

        plt.tight_layout()
        if len(title) > 0:
            plt.suptitle(title, fontsize=15, weight="bold", y=1.02)
        if save_path is not None:
            from pathlib import Path as _P
            _P(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"Figure écrite → {save_path}")
        if show:
            plt.show()
        plt.close()


def two_metrics_bounds(
    metric1, metric2, all_bounds_SR, all_bounds_MCD, num_labels=15, xlim=None, ylim=None
):
    """
    Compare two metric bounds (e.g., SR vs MCD) on a scatter plot.
    """

    # --- STYLE ONLY (no logic changes) -------------------------------------------
    rc = {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D0D0D0",
        "grid.alpha": 0.4,
    }
    with plt.rc_context(rc), plt.style.context("seaborn-v0_8-whitegrid"):
        plt.figure(figsize=(6, 4.5))
        ax = plt.gca()
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#B0B0B0")
            spine.set_linewidth(0.9)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
        # -----------------------------------------------------------------------------

        #### SR ####
        x1 = all_bounds_SR[metric1]
        y1 = all_bounds_SR[metric2]
        labels = list(zip(all_bounds_SR["thetas"], all_bounds_SR["coverages"]))
        plt.scatter(x=x1, y=y1, marker="+", label="SR", s=60, linewidths=1.8, zorder=2)

        # Choose evenly spaced indices along the curve
        indices = np.linspace(0, len(x1) - 1, num=num_labels, dtype=int)
        for j in indices:
            label = f"({labels[j][0]:.2f}, {labels[j][1]:.2f})"
            plt.annotate(
                label,
                (x1[j], y1[j]),
                textcoords="offset points",
                xytext=(0, -14),  # 👇 below the point
                ha="center",
                va="top",
                fontsize=10,
                zorder=3,
            )

        if all_bounds_MCD is not None:
            #### MCD ####
            x2 = all_bounds_MCD[metric1]
            y2 = all_bounds_MCD[metric2]
            plt.scatter(
                x=x2,
                y=y2,
                marker="^",
                label="MCD",
                c="y",
                s=40,
                edgecolors="k",
                linewidths=0.6,
                zorder=2,
            )
            # ❌ no annotations here

        if num_labels > 0:
            # Legend-like note for annotations
            plt.text(
                0.02,
                0.98,
                r"Ticks: ($\theta$, coverage)",
                transform=plt.gca().transAxes,
                fontsize=11,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
                zorder=4,
            )

        plt.xlabel(metric1 + " bound")
        plt.ylabel(metric2 + " bound")
        plt.legend(loc="best", borderpad=0.6, labelspacing=0.5, handlelength=2.0)
        plt.grid(True)

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        else:
            if all_bounds_MCD is not None:
                plt.xlim(min(min(x1), min(x2)) * 0.9, 1.1 * max(max(x1), max(x2)))
            else:
                plt.xlim(min(x1) * 0.9, 1.1 * max(x1))

        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        else:
            if all_bounds_MCD is not None:
                plt.ylim(min(min(y1), min(y2)) * 0.9, 1.1 * max(max(y1), max(y2)))
            else:
                plt.ylim(min(y1) * 0.9, 1.1 * max(y1))

        plt.tight_layout()
        plt.show()


def lines():
    """
    List of standardized plot parameters used in experiments notebooks
    """
    # Softmax Response (SR) by confidence threshold theta
    lines_list_1_SR = [
        {
            "kappa": "SR",
            "name": "Target",
            "colname": "metric_target",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#d51414",
            "style": "-",
        },
        {
            "kappa": "SR",
            "name": "Guaranteed",
            "colname": "metric_bound",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#1d3ae2",
            "style": "--",
        },
        {
            "kappa": "SR",
            "name": "On train set",
            "colname": "train_metric",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#8F7A1C",
            "style": "--",
        },
        {
            "kappa": "SR",
            "name": "On test set",
            "colname": "test_metric",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#0f770a",
            "style": "-",
        },
    ]
    # Softmax Response (SR) by coverage
    lines_list_2_SR = [
        {
            "kappa": "SR",
            "name": "Target",
            "colname": "metric_target",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#d51414",
            "style": "-",
        },
        {
            "kappa": "SR",
            "name": "Guaranteed",
            "colname": "metric_bound",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#1d3ae2",
            "style": "--",
        },
        {
            "kappa": "SR",
            "name": "On train set",
            "colname": "train_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#8F7A1C",
            "style": "--",
        },
        {
            "kappa": "SR",
            "name": "On test set",
            "colname": "test_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#0f770a",
            "style": "-",
        },
    ]
    # MCD minus variance by confidence threshold theta
    lines_list_1_MCD = [
        {
            "kappa": "MCD",
            "name": "Target",
            "colname": "metric_target",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#d51414",
            "style": "-",
        },
        {
            "kappa": "MCD",
            "name": "Guaranteed",
            "colname": "metric_bound",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#1d3ae2",
            "style": "--",
        },
        {
            "kappa": "MCD",
            "name": "On train set",
            "colname": "train_metric",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#8F7A1C",
            "style": "--",
        },
        {
            "kappa": "MCD",
            "name": "On test set",
            "colname": "test_metric",
            "x_axis": "theta_star",
            "x_axis_name": r"$\theta$",
            "colour": "#0f770a",
            "style": "-",
        },
    ]
    # MCD minus variance by coverage
    lines_list_2_MCD = [
        {
            "kappa": "MCD",
            "name": "Target",
            "colname": "metric_target",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#d51414",
            "style": "-",
        },
        {
            "kappa": "MCD",
            "name": "Guaranteed",
            "colname": "metric_bound",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#1d3ae2",
            "style": "--",
        },
        {
            "kappa": "MCD",
            "name": "On train set",
            "colname": "train_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#8F7A1C",
            "style": "--",
        },
        {
            "kappa": "MCD",
            "name": "On test set",
            "colname": "test_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#0f770a",
            "style": "-",
        },
    ]
    # SR and MCD on same plot, by coverage
    lines_list_2 = [
        {
            "kappa": "SR",
            "name": "Guaranteed (SR)",
            "colname": "metric_bound",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#053379",
            "style": "-",
        },
        {
            "kappa": "SR",
            "name": "On test set (SR)",
            "colname": "test_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#053379",
            "style": "--",
        },
        {
            "kappa": "MCD",
            "name": "Guaranteed (MCD)",
            "colname": "metric_bound",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#0f770a",
            "style": "-",
        },
        {
            "kappa": "MCD",
            "name": "On test set (MCD)",
            "colname": "test_metric",
            "x_axis": "test_coverage",
            "x_axis_name": "Coverage",
            "colour": "#0f770a",
            "style": "--",
        },
    ]

    return (
        lines_list_1_SR,
        lines_list_2_SR,
        lines_list_1_MCD,
        lines_list_2_MCD,
        lines_list_2,
    )
