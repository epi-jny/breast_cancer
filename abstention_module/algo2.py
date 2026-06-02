"""
Algo 2 — métriques CONDITIONNELLES par recherche GLOUTONNE (greedy).

Transposition fidèle de la section 2 du notebook : une figure par métrique,
par fonction de confiance, par axe (PAS de panel multi-métriques illisible).

Pour chaque métrique (FPR, FNR, PPV, SE, SP), avec ses propres `metric_targets`
et son propre `seed`, on produit via `sgp_metric_block` :
    sgp_<slug>_SR_vs_theta.png          SR vs θ            (Target/Guaranteed/train/test)
    sgp_<slug>_SR_vs_coverage.png       SR vs coverage
    sgp_<slug>_MCD_vs_theta.png         MCD vs θ           (si sgp_set_mcd.pkl présent)
    sgp_<slug>_MCD_vs_coverage.png      MCD vs coverage
    sgp_<slug>_SR_vs_MCD_coverage.png   SR v. MCD          (si SR et MCD)

PPV/SE/SP sont des bornes INFÉRIEURES (sens ">") : ylim adapté en conséquence.

⚠️ MCD nécessite un modèle avec dropout (SmallCNN OK, GMIC non).

Usage :
    uv run python -m abstention_module.algo2
    uv run python -m abstention_module.algo2 <nom_du_run>
"""

from abstention_module.nb_setup import *

# ── Config ───────────────────────────────────────────────────────────────────
RUN_DIR = Path("abstention_module/runs/cancer__gmic-nyu-sample1__20260529-132720")
#   abstention_module/runs/normalite__smallcnn__20260427-162128   # a du dropout (MCD OK)
#   abstention_module/runs/latest
if len(sys.argv) > 1:
    RUN_DIR = Path("abstention_module/runs") / sys.argv[1]

num_targets = 50
meta = json.loads((RUN_DIR / "meta.json").read_text())
theta_SR = (0.5, meta["kappa_max"])
theta_MCD = (-0.05, 0.0)
MODE = "greedy"   # Algo 2 = balayage glouton

# (label y, métrique, metric_targets, seed du split, sens du bound, slug fichier)
BLOCKS = [
    ("FPR", "FPR", np.linspace(0.01, 0.20, num_targets), 3, "<", "FPR"),
    ("FNR", "FNR", np.linspace(0.10, 0.35, num_targets), 4, "<", "FNR"),
    ("PPV", "PPV", np.linspace(0.25, 0.60, num_targets), 5, ">", "PPV"),
    ("SE",  "SE",  np.linspace(0.60, 0.90, num_targets), 6, ">", "SE"),
    ("SP",  "SP",  np.linspace(0.80, 0.99, num_targets), 7, ">", "SP"),
]

sgp_SR = pickle.load(open(RUN_DIR / "sgp_set.pkl", "rb"))
mcd_path = RUN_DIR / "sgp_set_mcd.pkl"
sgp_MCD = pickle.load(open(mcd_path, "rb")) if mcd_path.exists() else None
if sgp_MCD is None:
    print(f"[MCD] pas de {mcd_path.name} → seules les figures SR sont produites.")

for label, metric, targets, seed, direction, slug in BLOCKS:
    train_SR, test_SR = train_test_split(sgp_SR, seed=seed)
    res_SR = sgp_at_targets(
        train_SR, test_SR, metric_targets=targets, metric=metric, mode=MODE,
        theta_min=theta_SR[0], theta_max=theta_SR[1],
    )
    all_results = {"SR": res_SR}

    if sgp_MCD is not None:
        train_MCD, test_MCD = train_test_split(sgp_MCD, seed=seed)
        all_results["MCD"] = sgp_at_targets(
            train_MCD, test_MCD, metric_targets=targets, metric=metric, mode=MODE,
            theta_min=theta_MCD[0], theta_max=theta_MCD[1],
        )

    if res_SR.empty:
        print(f"[{label}] SR vide pour targets [{targets.min():.3f}, {targets.max():.3f}] — sauté "
              f"(typique en fort déséquilibre : peu de positifs).")
        continue

    sgp_metric_block(all_results, label, slug, RUN_DIR, theta_SR, theta_MCD, direction=direction)
    print(f"[{label}] ✓  ({len(res_SR)} cibles atteintes, "
          f"coverage∈[{res_SR.test_coverage.min():.2f}, {res_SR.test_coverage.max():.2f}])")

print(f"\n✓ Figures écrites dans {RUN_DIR}/")
