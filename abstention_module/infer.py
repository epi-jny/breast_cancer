"""
Inférence pilotée par runs_config.yaml — produit le `sgp_set` pour SGP.

Le YAML ne contient qu'une liste de chemins vers `best.pt`. Tout le reste est
auto-détecté à partir du checkpoint et de son `args.json` voisin :

  - dataset       : 'rsna' (mammographies) ou 'cifar' (CIFAR-10 binaire)
                    déduit du champ `target` de args.json
                    (target.startswith('cifar') → CIFAR, sinon RSNA)
  - in_channels   : lu directement dans conv1.weight du state_dict
                    (RSNA grayscale = 1, CIFAR RGB = 3)
  - stats         : grayscale (RSNA 1ch) ou rgb (RSNA 3ch dupliqués / CIFAR natif)
  - img_size      : lu dans args.json (512 pour RSNA, 96 par défaut pour CIFAR)
  - positive_class: pour CIFAR uniquement, lu dans args.json (0..9)
  - run_id        : <target>/<model_tag>/<timestamp>  (arborescence imbriquée,
                    calquée sur fine_tuning/checkpoints/runs/)
                    ex: normalite/smallcnn-focal/20260519-093143
                    ex: cifar_airplane_vs_all/smallcnn-bce-pos-weight/20260522-100539

Pour chaque checkpoint :
  - Si `runs/<run_id>/sgp_set.pkl` existe → skip (sauf --force)
  - Sinon → inférence sur le set cal+test, écrit `sgp_set.{pkl,csv}` + `meta.json`

Le lien `runs/latest` pointe toujours vers le dernier run écrit.

Usage:
    python -m abstention_module.infer                                  # toute la config
    python -m abstention_module.infer --only normalite/smallcnn-focal  # une seule
    python -m abstention_module.infer --force                          # réécrit tout
    python -m abstention_module.infer --ckpt <path>                    # mode ad-hoc
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from abstention_module.dataset import (
    PROJECT_ROOT,
    CIFARUnseenDataset,
    UnseenDataset,
    build_cifar_unseen_dataframe,
    build_unseen_dataframe,
    build_unseen_dataframe_density,
    load_stats,
)

CONFIG_PATH = Path(__file__).parent / "runs_config.yaml"
RUNS_DIR = Path(__file__).parent / "runs"

# Noms des classes density (1-vs-all). Sert au libellé du run (model_tag) et au meta.
DENSITY_CLASS_NAMES = ["A", "B", "C", "D"]


class SmallCNN(nn.Module):
    """Architecture du SmallCNN — paramétrable en in_channels pour matcher
    les checkpoints RSNA (1 canal grayscale) ou CIFAR (3 canaux RGB)."""

    def __init__(self, num_classes: int = 1, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ─── Helpers d'auto-détection ────────────────────────────────────────────────

def _load_train_args(ckpt_path: Path) -> dict:
    candidate = ckpt_path.parent / "args.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text())
        except Exception:
            return {}
    return {}


def _read_state_dict(ckpt_path: Path) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state["state_dict"] if "state_dict" in state else state


def _detect_in_channels(ckpt_path: Path) -> int:
    """Lit conv1.weight dans le state_dict pour déterminer les canaux d'entrée."""
    return int(_read_state_dict(ckpt_path)["conv1.weight"].shape[1])


def _detect_arch(ckpt_path: Path) -> str:
    """'smallcnn' ou 'resnet18' selon les clés du state_dict."""
    sd = _read_state_dict(ckpt_path)
    if "layer1.0.conv1.weight" in sd:
        return "resnet18"
    if "fc1.weight" in sd and "fc2.weight" in sd:
        return "smallcnn"
    raise ValueError(f"Architecture non reconnue dans {ckpt_path} "
                     f"(clés: {list(sd.keys())[:5]}...)")


def _detect_num_classes(ckpt_path: Path) -> int:
    """Nombre de classes en sortie. Lu dans la dernière Linear."""
    sd = _read_state_dict(ckpt_path)
    if "fc2.weight" in sd:               # SmallCNN
        return int(sd["fc2.weight"].shape[0])
    if "fc.1.weight" in sd:              # ResNet18 (head Sequential(Dropout, Linear))
        return int(sd["fc.1.weight"].shape[0])
    if "fc.weight" in sd:                # ResNet18 (head plat)
        return int(sd["fc.weight"].shape[0])
    raise ValueError(f"Tête non reconnue dans {ckpt_path}")


def _detect_dataset(ckpt_path: Path) -> str:
    """Retourne 'cifar', 'rsna_density' ou 'rsna' en fonction du `target` lu dans args.json."""
    args = _load_train_args(ckpt_path)
    target = str(args.get("target", "")).lower()
    dataset_name = str(args.get("dataset_name", "")).lower()
    if target.startswith("cifar") or "cifar" in dataset_name:
        return "cifar"
    if target == "density":
        return "rsna_density"
    return "rsna"


def _derive_run_id(ckpt_path: Path) -> tuple[str, str, str]:
    """Composantes (target, model_tag, timestamp) du run, calquées sur la
    structure des checkpoints fine_tuning .../runs/<target>/<model_tag>/<timestamp>/best.pt.

    Lit le `target` depuis args.json (plus fiable) si dispo, sinon fallback sur
    le segment de chemin correspondant.

    Le run d'abstention reprend la même arborescence imbriquée :
        runs/<target>/<model_tag>/<timestamp>/
    Ex: runs/normalite/smallcnn-focal/20260519-093143
        runs/cifar_airplane_vs_all/smallcnn-bce-pos-weight/20260522-100539
    """
    timestamp = ckpt_path.parent.name
    model_tag = ckpt_path.parent.parent.name
    args = _load_train_args(ckpt_path)
    target = args.get("target") or ckpt_path.parent.parent.parent.name
    return target, model_tag, timestamp


def _class_label(dataset_kind: str, positive_class: int) -> str:
    """Étiquette lisible de la classe positive, suffixée au model_tag en multi-classe.

    density → lettre de classe A/B/C/D (1-vs-all) ; sinon fallback générique 'posN'.
    Ex: resnet18_pretrained-D  (au lieu de resnet18_pretrained-pos3).
    """
    if dataset_kind == "rsna_density" and 0 <= positive_class < len(DENSITY_CLASS_NAMES):
        return DENSITY_CLASS_NAMES[positive_class]
    return f"pos{positive_class}"


# ─── Modèle + DataLoader ─────────────────────────────────────────────────────

def _build_resnet18(num_classes: int) -> nn.Module:
    """Reproduit l'archi de `fine_tuning/train_resnet_density.py:build_resnet18`."""
    model = models.resnet18(weights=None)
    model.avgpool = nn.Sequential(
        nn.Dropout2d(p=0.3),
        model.avgpool,
    )
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _load_model(ckpt_path: Path, device: torch.device, arch: str, in_channels: int, num_classes: int) -> nn.Module:
    sd = _read_state_dict(ckpt_path)
    if arch == "smallcnn":
        model = SmallCNN(num_classes=num_classes, in_channels=in_channels)
    elif arch == "resnet18":
        model = _build_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"arch inconnue: {arch}")
    model = model.to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def _build_loader_rsna_density(positive_class: int, batch_size: int, num_workers: int, device: torch.device):
    """Loader RSNA pour 1-vs-all sur density. Le `label` est binaire (class == positive_class)."""
    df = build_unseen_dataframe_density(positive_class=positive_class)
    df = df.rename(columns={"label": "normalite"})  # UnseenDataset attend la colonne 'normalite'
    mean, std = load_stats()  # stats RGB classiques

    dataset = UnseenDataset(df, mean=mean, std=std)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    info = {
        "dataset_kind": "rsna_density",
        "image_dir": str((PROJECT_ROOT / "data" / "preprocess_image" / "rsna_output" / "cropped_512").relative_to(PROJECT_ROOT)),
        "stats_mean": mean,
        "stats_std": std,
        "stats_choice": "rgb",
        "img_size": 512,
        "positive_class": positive_class,
        "positive_class_name": DENSITY_CLASS_NAMES[positive_class],
    }
    return loader, df.rename(columns={"normalite": "label"}), info


def _build_loader_rsna(in_channels: int, batch_size: int, num_workers: int, device: torch.device):
    df = build_unseen_dataframe()
    stats = "grayscale" if in_channels == 1 else "rgb"

    if stats == "grayscale":
        stats_file = PROJECT_ROOT / "fine_tuning" / "checkpoints" / "train_stats_512_grayscale.json"
        payload = json.loads(stats_file.read_text())
        mean, std = payload["mean"], payload["std"]
    else:
        mean, std = load_stats()

    dataset = UnseenDataset(df, mean=mean, std=std)
    if in_channels == 1:
        dataset.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean[:1], std[:1]),
        ])

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    info = {
        "dataset_kind": "rsna",
        "image_dir": str((PROJECT_ROOT / "data" / "preprocess_image" / "rsna_output" / "cropped_512").relative_to(PROJECT_ROOT)),
        "stats_mean": mean,
        "stats_std": std,
        "stats_choice": stats,
        "img_size": 512,
    }
    return loader, df, info


def _build_loader_cifar(ckpt_args: dict, batch_size: int, num_workers: int, device: torch.device):
    pos_class = ckpt_args.get("positive_class")
    img_size = int(ckpt_args.get("img_size", 96))
    if pos_class is None:
        raise ValueError(
            "Checkpoint CIFAR sans `positive_class` dans args.json. "
            "Le run a probablement été créé avant le refactor — relance l'entraînement."
        )

    df = build_cifar_unseen_dataframe(positive_class=pos_class)
    df = df.rename(columns={"label": "y_label"})  # éviter conflit dans le merge final

    dataset = CIFARUnseenDataset(df.rename(columns={"y_label": "label"}), positive_class=pos_class, img_size=img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    info = {
        "dataset_kind": "cifar",
        "image_dir": "data/raw/cifar10 (torchvision CIFAR-10 test set)",
        "stats_mean": [0.4914, 0.4822, 0.4465],
        "stats_std": [0.2470, 0.2435, 0.2616],
        "stats_choice": "cifar_native_rgb",
        "img_size": img_size,
        "positive_class": pos_class,
        "positive_class_name": ckpt_args.get("positive_class_name"),
    }
    return loader, df.rename(columns={"y_label": "label"}), info


# ─── Inférence ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(model, loader, device, positive_class: int | None = None) -> pd.DataFrame:
    """Inférence — produit (image_id, y_true, y_pred, p, kappa) par image.

    - Binaire (num_classes=1) : p = sigmoid(logit), positive_class ignoré.
    - Multi-classe + positive_class : p = softmax(logits)[:, positive_class] (1-vs-all).
    """
    rows = []
    for x, y, image_id in tqdm(loader, desc="Inference", leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        if out.dim() == 2 and out.shape[1] > 1:
            if positive_class is None:
                raise ValueError("Modèle multi-classe sans `positive_class` — ajoute-le dans le YAML.")
            p = torch.softmax(out, dim=1)[:, positive_class].cpu()
        else:
            logit = out.squeeze(1) if out.dim() > 1 else out
            p = torch.sigmoid(logit).cpu()
        y_pred = (p >= 0.5).long()
        kappa = torch.maximum(p, 1.0 - p)
        ids = image_id.tolist() if isinstance(image_id, torch.Tensor) else list(image_id)
        for iid, yt, yp, prob, k in zip(
            ids, y.tolist(), y_pred.tolist(), p.tolist(), kappa.tolist()
        ):
            rows.append({"image_id": int(iid), "y_true": int(yt), "y_pred": int(yp), "p": float(prob), "kappa": float(k)})
    return pd.DataFrame(rows)


def _abstention_diagnostic(sgp: pd.DataFrame) -> dict:
    """Évalue si le modèle est utilisable pour l'abstention sélective.

    Critères de pathologie (sur le set complet cal+test) :
      - `pred_collapse`     : pred_positive_rate ∈ [0, 0.02] ∪ [0.98, 1] → prédicteur constant
      - `kappa_compressed`  : kappa_max - kappa_min < 0.10 → pas de signal de confiance
      - `ranking_inverted`  : accuracy parmi les samples haute-confiance (top 10%) < parmi les low-conf →
                              la SR ordonne mal les exemples (effondrement type bce-pos-weight)
    """
    flags = []
    pred_pos = float(sgp["y_pred"].mean())
    kappa_min = float(sgp["kappa"].min())
    kappa_max = float(sgp["kappa"].max())
    kappa_spread = kappa_max - kappa_min

    if pred_pos <= 0.02 or pred_pos >= 0.98:
        flags.append(f"pred_collapse(pred_positive_rate={pred_pos:.3f})")
    if kappa_spread < 0.10:
        flags.append(f"kappa_compressed(spread={kappa_spread:.3f})")

    # Test ranking : la SR doit faire monter l'accuracy quand on filtre les samples confiants
    n_top = max(50, int(0.10 * len(sgp)))
    top = sgp.nlargest(n_top, "kappa")
    bot = sgp.nsmallest(n_top, "kappa")
    acc_top = float((top.y_true == top.y_pred).mean())
    acc_bot = float((bot.y_true == bot.y_pred).mean())
    if acc_top < acc_bot:
        flags.append(f"ranking_inverted(acc_top10%={acc_top:.3f} < acc_bot10%={acc_bot:.3f})")

    return {
        "usable_for_abstention": len(flags) == 0,
        "pathology_flags": flags,
        "pred_positive_rate_global": round(pred_pos, 4),
        "kappa_spread": round(kappa_spread, 4),
        "accuracy_top10pct_kappa": round(acc_top, 4),
        "accuracy_bot10pct_kappa": round(acc_bot, 4),
    }


def _summary_stats(sgp: pd.DataFrame) -> dict:
    out = {}
    for split in ["cal", "test"]:
        s = sgp[sgp["split"] == split]
        if len(s) == 0:
            continue
        out[split] = {
            "n": int(len(s)),
            "positives": int(s["y_true"].sum()),
            "base_rate": round(float(s["y_true"].mean()), 4),
            "accuracy_at_0.5": round(float((s["y_true"] == s["y_pred"]).mean()), 4),
            "pred_positive_rate": round(float(s["y_pred"].mean()), 4),
            "p_mean": round(float(s["p"].mean()), 4),
            "p_std": round(float(s["p"].std()), 4),
            "kappa_mean": round(float(s["kappa"].mean()), 4),
            "kappa_min": round(float(s["kappa"].min()), 4),
            "kappa_max": round(float(s["kappa"].max()), 4),
        }
    return out


def process_checkpoint(ckpt: Path, force: bool, batch_size: int, num_workers: int, device: torch.device, positive_class_override: int | None = None) -> str:
    """Traite un checkpoint (chemin). Tout est auto-détecté.
    `positive_class_override` : pour les modèles multi-classes, override depuis le YAML.
    Retourne 'skipped', 'done' ou 'failed'."""
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        print(f"[ERR ]  checkpoint introuvable : {ckpt}")
        return "failed"

    in_channels = _detect_in_channels(ckpt)
    arch = _detect_arch(ckpt)
    num_classes = _detect_num_classes(ckpt)
    dataset_kind = _detect_dataset(ckpt)
    train_args = _load_train_args(ckpt)
    target = train_args.get("target", "unknown")
    dataset_name = train_args.get("dataset_name", "unknown")

    # Résoudre positive_class : YAML override > args.json (CIFAR) > None
    positive_class = positive_class_override
    if positive_class is None and num_classes == 1:
        positive_class = train_args.get("positive_class")  # CIFAR le stocke ici
    if num_classes > 1 and positive_class is None:
        print(f"[ERR ]  {ckpt.name} : modèle {num_classes}-classes mais positive_class non fourni "
              f"(à mettre dans le YAML).")
        return "failed"

    # Composantes du run : pour multi-classe, suffixer le model_tag par la classe
    # positive (density → A/B/C/D ; fallback générique posN).
    target_part, model_tag, ts = _derive_run_id(ckpt)
    if num_classes > 1:
        model_tag = f"{model_tag}-{_class_label(dataset_kind, positive_class)}"

    # Arborescence imbriquée runs/<target>/<model_tag>/<timestamp>/ (cf. fine_tuning)
    out_dir = RUNS_DIR / target_part / model_tag / ts
    name = str(out_dir.relative_to(RUNS_DIR))   # id du run, ex "normalite/smallcnn/20260427-162128"
    sgp_path = out_dir / "sgp_set.pkl"
    meta_path = out_dir / "meta.json"

    if sgp_path.exists() and meta_path.exists() and not force:
        print(f"[SKIP]  {name}  →  déjà calculé ({sgp_path})")
        return "skipped"

    print(f"[RUN ]  {name}")
    print(f"        target     : {target}")
    print(f"        dataset    : {dataset_name}  (kind={dataset_kind})")
    print(f"        arch       : {arch}  (num_classes={num_classes}, in_channels={in_channels})")
    if positive_class is not None and num_classes > 1:
        print(f"        1-vs-all   : positive_class={positive_class}")
    print(f"        ckpt       : {ckpt.relative_to(PROJECT_ROOT) if ckpt.is_relative_to(PROJECT_ROOT) else ckpt}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset_kind == "cifar":
        loader, df, info = _build_loader_cifar(train_args, batch_size, num_workers, device)
    elif dataset_kind == "rsna_density":
        loader, df, info = _build_loader_rsna_density(positive_class, batch_size, num_workers, device)
    else:
        loader, df, info = _build_loader_rsna(in_channels, batch_size, num_workers, device)

    t0 = time.time()
    model = _load_model(ckpt, device, arch=arch, in_channels=in_channels, num_classes=num_classes)
    preds = _run_inference(model, loader, device, positive_class=positive_class if num_classes > 1 else None)
    duration = time.time() - t0

    sgp = df[["image_id", "split"]].merge(preds, on="image_id", how="inner")
    sgp.to_pickle(sgp_path)
    sgp.to_csv(out_dir / "sgp_set.csv", index=False)

    diag = _abstention_diagnostic(sgp)

    meta = {
        "name": name,
        "target": target,
        "dataset_name": dataset_name,
        "dataset_kind": dataset_kind,
        "model_tag": ckpt.parent.parent.name,
        "training_timestamp": ckpt.parent.name,
        "abstention_diagnostic": diag,
        "kappa_min": round(float(sgp["kappa"].min()), 6),
        "kappa_max": round(float(sgp["kappa"].max()), 6),
        "inference": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(duration, 1),
            "device": str(device),
            "batch_size": batch_size,
            "in_channels": in_channels,
            **info,
        },
        "checkpoint": {
            "path": str(ckpt.relative_to(PROJECT_ROOT)) if ckpt.is_relative_to(PROJECT_ROOT) else str(ckpt),
            "best_auc": train_args.get("best_auc"),
            "best_epoch": train_args.get("best_epoch"),
            "loss": train_args.get("loss"),
            "sampler": train_args.get("sampler"),
        },
        "training_args": train_args,
        "dataset": {
            "n_total": int(len(sgp)),
            "split_sizes": {k: int(v) for k, v in sgp["split"].value_counts().to_dict().items()},
        },
        "baselines": _summary_stats(sgp),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    latest = RUNS_DIR / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(name, target_is_directory=True)

    print(f"        écrit : {sgp_path}  ({len(sgp)} lignes, {duration:.0f}s)")
    for split, m in meta["baselines"].items():
        print(f"        {split:>4} : n={m['n']:>5}  pos={m['positives']:>4} ({m['base_rate']*100:.1f}%)  "
              f"acc@0.5={m['accuracy_at_0.5']*100:5.2f}%  "
              f"pred_pos={m['pred_positive_rate']*100:5.2f}%  "
              f"kappa∈[{m['kappa_min']:.3f},{m['kappa_max']:.3f}]")
    if diag["usable_for_abstention"]:
        print(f"        ✅ Utilisable pour abstention "
              f"(pred_pos={diag['pred_positive_rate_global']:.2f}, "
              f"κ_spread={diag['kappa_spread']:.2f}, "
              f"acc top10%/bot10%={diag['accuracy_top10pct_kappa']:.2f}/{diag['accuracy_bot10pct_kappa']:.2f})")
    else:
        print(f"        ⚠️  NON utilisable pour abstention : {', '.join(diag['pathology_flags'])}")
    return "done"


def load_config(path: Path) -> list[tuple[Path, dict]]:
    """Parse le YAML. Chaque entrée peut être :
      - une string (chemin vers best.pt, options par défaut)
      - un dict {path: ..., positive_class: N, ...} pour spécifier des options
    Retourne une liste de (path, options).
    """
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("checkpoints", []) if isinstance(raw, dict) else []
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append((Path(e), {}))
        elif isinstance(e, dict) and "path" in e:
            opts = {k: v for k, v in e.items() if k != "path"}
            out.append((Path(e["path"]), opts))
        else:
            raise ValueError(f"Entrée YAML invalide : {e!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG_PATH,
                    help=f"YAML listant les checkpoints (défaut: {CONFIG_PATH.name})")
    ap.add_argument("--only", type=str, default=None,
                    help="Ne traite que le run dont le nom déduit matche (substring)")
    ap.add_argument("--force", action="store_true",
                    help="Réécrit même si runs/<name>/sgp_set.pkl existe déjà")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Mode ad-hoc : un chemin précis (ignore la config)")
    ap.add_argument("--positive-class", type=int, default=None,
                    help="Pour modèles multi-classes : index de la classe positive en mode 1-vs-all "
                         "(ex: 3 pour D dans density A/B/C/D). Override le YAML.")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    if args.ckpt is not None:
        process_checkpoint(args.ckpt, args.force, args.batch_size, args.num_workers, device,
                           positive_class_override=args.positive_class)
        return

    if not args.config.exists():
        print(f"Config introuvable : {args.config}")
        return

    entries = load_config(args.config)
    if args.only:
        entries = [(p, o) for p, o in entries
                   if args.only in "/".join(_derive_run_id(p if p.is_absolute() else PROJECT_ROOT / p))]
        if not entries:
            print(f"Aucun checkpoint dont le nom contient {args.only!r}")
            return

    print(f"Checkpoints à traiter : {len(entries)}")
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for p, opts in entries:
        status = process_checkpoint(p, args.force, args.batch_size, args.num_workers, device,
                                    positive_class_override=opts.get("positive_class"))
        counts[status] += 1
        print()

    print(f"Bilan : {counts['done']} run(s) effectué(s), {counts['skipped']} skip(s), {counts['failed']} échec(s)")


if __name__ == "__main__":
    main()
