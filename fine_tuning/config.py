"""
Configuration centrale du fine-tuning.
Modifier ce fichier pour changer les chemins et hyperparamètres.
"""

import os

# ─── Chemins ────────────────────────────────────────────────────────────────

# Dossier racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Nom du run de preprocessing à utiliser (dossier dans output/)
# Changer cette ligne pour pointer vers un autre run
RUN_NAME = "rsna_output"

# Dossier du run — produit par preprocess.py + inference.py
RUN_DIR = os.path.join(PROJECT_ROOT, "data", "preprocess_image", RUN_NAME)

# data.pkl : liste des exams avec leurs labels et chemins d'images
EXAM_LIST_PATH = os.path.join(RUN_DIR, "data.pkl")

# cropped_512/ : images uint8 512×512 pré-resizées par scripts/preresize_images.py
# (fallback sur cropped_images/ 2944×1920 si le cache 512 n'existe pas encore)
IMAGE_DIR = os.path.join(RUN_DIR, "cropped_512")

# Dossier où sauvegarder les checkpoints du modèle
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "fine_tuning", "checkpoints")

# ─── Split train / validation ────────────────────────────────────────────────

# Proportion des exams utilisée pour la validation (le reste = train)
VAL_SPLIT = 0.2

# Graine aléatoire pour reproductibilité du split
RANDOM_SEED = 42

# ─── Images ─────────────────────────────────────────────────────────────────

# Taille de redimensionnement (H, W) des images en entrée du modèle.
# Les originales font 2944x1920 — trop grand pour 4GB VRAM.
# Réduire ici si OOM (Out Of Memory).
IMAGE_SIZE = (1472, 960)   # moitié de la résolution originale

# ─── Entraînement ────────────────────────────────────────────────────────────

# Taille de batch (1 exam = 4 images). Avec 4GB VRAM, garder à 1 ou 2.
BATCH_SIZE = 1

# Nombre d'epochs
NUM_EPOCHS = 50

# Taux d'apprentissage initial
LEARNING_RATE = 1e-4

# Poids L2 (régularisation)
WEIGHT_DECAY = 1e-5

# Nombre de workers pour le DataLoader (0 = pas de multiprocessing)
NUM_WORKERS = 8

# Device d'entraînement. "cuda" si GPU disponible, sinon "cpu".
DEVICE = "cuda"
