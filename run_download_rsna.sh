#!/usr/bin/env bash
# Telecharge + extrait le dataset complet RSNA Breast Cancer sur la VM.
# Auth : ~/.kaggle/access_token (token KGAT, kaggle 2.x).
# A lancer en tmux (plusieurs heures) :
#   tmux new-session -d -s dl "bash ~/breast_cancer/run_download_rsna.sh 2>&1 | tee ~/breast_cancer/data/rsna/download.log"
set -euo pipefail

DEST="$HOME/breast_cancer/data/rsna"
PY="$HOME/breast_cancer/.venv/bin"
ZIP="rsna-breast-cancer-detection.zip"

mkdir -p "$DEST"
cd "$DEST"

echo "[$(date '+%F %T')] Espace disque avant download :"
df -h "$DEST"

echo "[$(date '+%F %T')] Telechargement du dataset complet..."
"$PY/kaggle" competitions download -c rsna-breast-cancer-detection -p "$DEST"

if [ ! -f "$ZIP" ]; then
  echo "[$(date '+%F %T')] ERREUR : $ZIP introuvable apres download." >&2
  exit 1
fi
echo "[$(date '+%F %T')] Download termine : $(du -h "$ZIP" | cut -f1)"

echo "[$(date '+%F %T')] Espace disque avant extraction :"
df -h "$DEST"

echo "[$(date '+%F %T')] Extraction..."
unzip -q "$ZIP" -d "$DEST"

echo "[$(date '+%F %T')] Extraction OK. Suppression du zip..."
rm -f "$ZIP"

echo "[$(date '+%F %T')] TERMINE. Contenu :"
ls -lh "$DEST"
df -h "$DEST"
