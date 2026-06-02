"""
Inférence GMIC sur un dataset préprocessé par preprocess.py.

Utilise ScratchGMIC (gmic_from_scratch.py) — pas de best_center requis.
Chargement direct des PNG avec imageio + normalisation z-score.

Sauvegarde incrémentale : chaque image est écrite dans le CSV dès qu'elle
est traitée. Le script peut être interrompu et relancé : les images déjà
présentes dans le CSV sont skippées.

Usage :
    uv run python scripts/inference.py --output-dir data/preprocess_image/rsna_output
    uv run python scripts/inference.py --output-dir data/preprocess_image/rsna_output --model-index 1
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))  # pour importer gmic_from_scratch

from gmic_from_scratch import ScratchGMIC, load_nyu_weights  # noqa: E402

VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]
FIELDNAMES = ["image_index", "malignant_pred", "benign_pred", "malignant_label"]

PERCENT_T = {"1": 0.02, "2": 0.03, "3": 0.03, "4": 0.05, "5": 0.1}


def load_done(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["image_index"])
    return done


def build_model(model_index: str, device: torch.device) -> ScratchGMIC:
    model = ScratchGMIC(
        K=6,
        crop_shape=(256, 256),
        cam_size=(46, 30),
        percent_t=PERCENT_T[model_index],
        num_classes=2,
        device_type="gpu" if device.type == "cuda" else "cpu",
        gpu_number=device.index or 0,
    ).to(device).eval()

    ckpt = _PROJECT_ROOT / "GMIC" / "models" / f"sample_model_{model_index}.p"
    load_nyu_weights(model, str(ckpt), device=device)
    return model


def run(exam_list, model, image_dir: str, csv_path: str, device: torch.device, done: set) -> int:
    write_header = not os.path.exists(csv_path)
    written = 0

    with open(csv_path, "a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with torch.no_grad():
            for datum in tqdm.tqdm(exam_list, desc="Exams"):
                for view in VIEWS:
                    images_for_view = datum.get(view, [])
                    if not images_for_view:
                        continue
                    short_path = images_for_view[0]

                    if short_path in done:
                        continue

                    img_file = os.path.join(image_dir, short_path + ".png")
                    if not os.path.exists(img_file):
                        tqdm.tqdm.write(f"[WARN] introuvable : {img_file}")
                        continue

                    try:
                        img = imageio.imread(img_file).astype(np.float32)
                        if img.shape != (2944, 1920):
                            img = cv2.resize(img, (1920, 2944), interpolation=cv2.INTER_AREA)
                        img = (img - img.mean()) / max(img.std(), 1e-5)
                        x = torch.tensor(img[None, None]).to(device)
                        y = model(x)
                    except Exception as e:
                        tqdm.tqdm.write(f"[WARN] erreur sur {short_path}: {type(e).__name__}: {e}")
                        continue

                    benign_pred = float(y[0, 0])
                    malignant_pred = float(y[0, 1])

                    cancer = datum["cancer_label"]
                    if view.startswith("L-"):
                        malignant_label = float(cancer.get("left_malignant", 0))
                    else:
                        malignant_label = float(cancer.get("right_malignant", 0))

                    writer.writerow({
                        "image_index": short_path,
                        "malignant_pred": f"{malignant_pred:.4f}",
                        "benign_pred": f"{benign_pred:.4f}",
                        "malignant_label": f"{malignant_label:.4f}",
                    })
                    done.add(short_path)
                    written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Dossier produit par preprocess.py (contient data.pkl et cropped_images/)")
    parser.add_argument("--model-index", default="1", choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    pkl_path = os.path.join(output_dir, "data.pkl")
    image_dir = os.path.join(output_dir, "cropped_images")
    csv_path = os.path.join(output_dir, "predictions.csv")

    if not os.path.exists(pkl_path):
        print(f"Erreur : data.pkl introuvable dans {output_dir}")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_number}")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    with open(pkl_path, "rb") as f:
        exam_list = pickle.load(f)
    print(f"Exams : {len(exam_list)}")

    done = load_done(csv_path)
    print(f"Déjà traités : {len(done)} images")

    model = build_model(args.model_index, device)

    written = run(exam_list, model, image_dir, csv_path, device, done)
    print(f"\nTerminé — {written} nouvelles prédictions → {csv_path}")


if __name__ == "__main__":
    main()
