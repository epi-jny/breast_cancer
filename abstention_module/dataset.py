"""
Dataset pour l'évaluation post-hoc de l'abstention.

Supporte deux pipelines :
  - RSNA mammographies (`build_unseen_dataframe`) : images sur disque non vues
    pendant l'entraînement (patients absents de `train_subset.csv`).
  - CIFAR-10 binaire (`build_cifar_unseen_dataframe`) : test set torchvision
    (10 000 images), relabel one-vs-all selon `positive_class`.

Dans les deux cas, on découpe en :
  - calibration (75 %) : sert à choisir le seuil θ*
  - test (25 %)        : sert à rapporter la courbe risk-coverage finale

Split déterministe (seed=42).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "preprocess_image" / "rsna_output" / "cropped_512"
FULL_CSV = PROJECT_ROOT / "data" / "raw" / "extract_dataset3" / "train.csv"
TRAIN_SUBSET_CSV = PROJECT_ROOT / "data" / "raw" / "rsna_images" / "train_subset.csv"
STATS_PATH = PROJECT_ROOT / "fine_tuning" / "checkpoints" / "train_stats_512.json"
CIFAR_DIR = PROJECT_ROOT / "data" / "raw" / "cifar10"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMG_SIZE = 512
RANDOM_SEED = 42
TEST_SIZE = 0.25  # 75 % cal / 25 % test → ratio 3:1


def _normalite_label(df: pd.DataFrame) -> pd.Series:
    cancer = df["cancer"].fillna(0).astype(int) == 1
    biopsy = df["biopsy"].fillna(0).astype(int) == 1
    dnc_raw = df["difficult_negative_case"]
    if dnc_raw.dtype == object:
        dnc = dnc_raw.astype(str).str.lower().isin(["true", "1"])
    else:
        dnc = dnc_raw.fillna(False).astype(bool)
    birads0 = df["BIRADS"].fillna(-1).astype(float) == 0
    return (cancer | biopsy | dnc | birads0).astype(int)


def build_unseen_dataframe() -> pd.DataFrame:
    """Retourne un DataFrame des images non vues avec label et split.

    Colonnes : patient_id, image_id, normalite, path, split ∈ {cal, test}.
    """
    full = pd.read_csv(FULL_CSV)
    subset = pd.read_csv(TRAIN_SUBSET_CSV)
    seen_patients = set(subset["patient_id"].astype(str).unique())
    disk_patients = set(os.listdir(IMAGE_DIR))

    full["pid_str"] = full["patient_id"].astype(str)
    unseen_pids = disk_patients - seen_patients
    df = full[full["pid_str"].isin(unseen_pids)].copy()

    df["path"] = df.apply(
        lambda r: str(IMAGE_DIR / r["pid_str"] / f"{r['image_id']}.png"),
        axis=1,
    )
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
    df["normalite"] = _normalite_label(df)

    patient_pos = df.groupby("patient_id")["normalite"].max().rename("any_pos").reset_index()
    cal_p, test_p = train_test_split(
        patient_pos,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=patient_pos["any_pos"],
    )
    cal_set = set(cal_p["patient_id"])
    df["split"] = df["patient_id"].map(lambda p: "cal" if p in cal_set else "test")

    return df[["patient_id", "image_id", "normalite", "path", "split"]]


def load_stats() -> tuple[list[float], list[float]]:
    stats = json.loads(STATS_PATH.read_text())
    return stats["mean"], stats["std"]


def build_unseen_dataframe_density(positive_class: int) -> pd.DataFrame:
    """Variante de `build_unseen_dataframe` pour les modèles density (4-classes).
    Relabel one-vs-all : label=1 si la density == positive_class, 0 sinon.

    positive_class ∈ {0,1,2,3} pour A/B/C/D respectivement.

    Colonnes : patient_id, image_id, label, path, split ∈ {cal, test}.
    """
    if positive_class not in {0, 1, 2, 3}:
        raise ValueError(f"positive_class doit être ∈ {{0,1,2,3}} (reçu: {positive_class})")

    full = pd.read_csv(FULL_CSV)
    subset = pd.read_csv(TRAIN_SUBSET_CSV)
    seen_patients = set(subset["patient_id"].astype(str).unique())
    disk_patients = set(os.listdir(IMAGE_DIR))

    full["pid_str"] = full["patient_id"].astype(str)
    unseen_pids = disk_patients - seen_patients
    df = full[full["pid_str"].isin(unseen_pids)].copy()
    df = df.dropna(subset=["density"])  # only images avec density renseignée

    density_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    df["density_int"] = df["density"].map(density_map)
    df = df.dropna(subset=["density_int"])
    df["density_int"] = df["density_int"].astype(int)
    df["label"] = (df["density_int"] == positive_class).astype(int)

    df["path"] = df.apply(
        lambda r: str(IMAGE_DIR / r["pid_str"] / f"{r['image_id']}.png"),
        axis=1,
    )
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)

    patient_pos = df.groupby("patient_id")["label"].max().rename("any_pos").reset_index()
    cal_p, test_p = train_test_split(
        patient_pos, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=patient_pos["any_pos"],
    )
    cal_set = set(cal_p["patient_id"])
    df["split"] = df["patient_id"].map(lambda p: "cal" if p in cal_set else "test")

    return df[["patient_id", "image_id", "label", "path", "split"]]


class UnseenDataset(Dataset):
    """Dataset image-level. Retourne (tensor_image, label, image_id)."""

    def __init__(self, df: pd.DataFrame, mean: list[float], std: list[float], img_size: int = IMG_SIZE):
        self.paths = df["path"].tolist()
        self.labels = df["normalite"].astype(int).tolist()
        self.image_ids = df["image_id"].astype(int).tolist()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        return (
            self.transform(img),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.image_ids[idx],
        )


# ─── CIFAR-10 binaire ────────────────────────────────────────────────────────

def build_cifar_unseen_dataframe(positive_class: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Retourne un DataFrame des 10 000 images du test set CIFAR-10, splitté
    en cal/test (75/25) avec relabel one-vs-all.

    Colonnes : image_id (indice dans test set), label (0/1), split ∈ {cal, test}.
    """
    base = CIFAR10(root=str(CIFAR_DIR), train=False, download=False)
    n = len(base)
    image_ids = np.arange(n)
    labels = np.array([1 if cls == positive_class else 0 for _, cls in base], dtype=int)

    cal_idx, test_idx = train_test_split(
        image_ids, test_size=TEST_SIZE, random_state=seed, stratify=labels,
    )
    split = np.empty(n, dtype=object)
    split[cal_idx] = "cal"
    split[test_idx] = "test"

    return pd.DataFrame({"image_id": image_ids, "label": labels, "split": split})


class CIFARUnseenDataset(Dataset):
    """Dataset CIFAR-10 binarisé pour inférence d'abstention.

    Retourne (tensor_image, label, image_id). Pas d'augmentation.
    """

    def __init__(self, df: pd.DataFrame, positive_class: int, img_size: int):
        self.base = CIFAR10(root=str(CIFAR_DIR), train=False, download=False)
        self.image_ids = df["image_id"].astype(int).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.positive_class = positive_class
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        iid = self.image_ids[idx]
        img, _cls = self.base[iid]  # PIL image
        return (
            self.transform(img),
            torch.tensor(self.labels[idx], dtype=torch.long),
            iid,
        )
