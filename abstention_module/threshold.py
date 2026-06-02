"""
Courbe accuracy = f(seuil de softmax response), reproduction du graphique
El-Yaniv (Geifman & El-Yaniv 2017, Selective Classification for Deep Neural
Networks).

Pour chaque seuil tau dans [kappa.min, kappa.max] :
    selected = {images où kappa >= tau}
    accuracy = (y_true == y_pred).mean() sur selected

Évalué par défaut sur le set de calibration (le plus grand).

Usage:
    python -m abstention_module.threshold
    python -m abstention_module.threshold --split test
    python -m abstention_module.threshold --num-thresholds 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).parent / "runs"


def _available_runs() -> list[str]:
    """Ids des runs disponibles, au format imbriqué <target>/<model_tag>/<timestamp>.

    Un run = un dossier contenant `meta.json` dans l'arborescence
    runs/<target>/<model_tag>/<timestamp>/ (le symlink `latest` est exclu)."""
    return sorted(
        str(meta.parent.relative_to(RUNS_DIR))
        for meta in RUNS_DIR.glob("*/*/*/meta.json")
    )


def _resolve_run_dir(run: str | None) -> Path:
    """Retourne le dossier du run choisi (par id imbriqué) ou `latest` par défaut."""
    if run is None:
        latest = RUNS_DIR / "latest"
        if not latest.exists():
            raise FileNotFoundError(
                f"Aucun lien {latest} — lance d'abord `python -m abstention_module.infer`."
            )
        return latest.resolve()
    candidate = RUNS_DIR / run
    if not candidate.exists():
        raise FileNotFoundError(
            f"Run introuvable : {candidate}\n"
            f"Runs disponibles : {_available_runs()}"
        )
    return candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Id du run imbriqué <target>/<model_tag>/<timestamp> dans "
                         "abstention_module/runs/ (défaut: latest)")
    ap.add_argument("--sgp", type=Path, default=None,
                    help="Mode ad-hoc : chemin direct vers un sgp_set.pkl (court-circuite --run)")
    ap.add_argument("--split", choices=["cal", "test", "all"], default="cal",
                    help="Sur quel split tracer la courbe (defaut: cal)")
    ap.add_argument("--num-thresholds", type=int, default=50)
    args = ap.parse_args()

    if args.sgp is not None:
        sgp_path = args.sgp
        out_plot = args.sgp.parent / "accuracy_vs_sr_threshold.png"
    else:
        run_dir = _resolve_run_dir(args.run)
        sgp_path = run_dir / "sgp_set.pkl"
        out_plot = run_dir / "accuracy_vs_sr_threshold.png"
        run_label = run_dir.relative_to(RUNS_DIR) if run_dir.is_relative_to(RUNS_DIR) else run_dir.name
        print(f"Run : {run_label}")

    sgp = pd.read_pickle(sgp_path)
    if args.split != "all":
        sgp = sgp[sgp["split"] == args.split].reset_index(drop=True)
    print(f"Set évalué : {args.split}  ({len(sgp)} images)")
    print(f"Accuracy baseline (tau=0)  : {(sgp.y_true == sgp.y_pred).mean()*100:.2f}%")

    thresholds = np.linspace(sgp.kappa.min(), sgp.kappa.max(), num=args.num_thresholds)
    accs = []
    coverages = []
    for t in thresholds:
        selected = sgp.loc[sgp.kappa >= t]
        cov = len(selected) / len(sgp)
        acc = (selected.y_true == selected.y_pred).mean() if len(selected) > 0 else np.nan
        accs.append(acc)
        coverages.append(cov)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accs, marker="o", markersize=3)
    plt.ylabel("Accuracy")
    plt.xlabel("SR threshold (kappa)")
    plt.title(f"Accuracy vs softmax-response threshold — split={args.split}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=120)
    print(f"\nPlot écrit → {out_plot}")

    print("\nQuelques points clés :")
    for cov_target in [1.0, 0.75, 0.50, 0.25, 0.10]:
        idx = np.argmin(np.abs(np.array(coverages) - cov_target))
        print(f"  coverage ≈ {coverages[idx]*100:5.1f}%  →  tau = {thresholds[idx]:.4f}  →  accuracy = {accs[idx]*100:.2f}%")


if __name__ == "__main__":
    main()
