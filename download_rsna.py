#!/usr/bin/env python
"""Telecharge le dataset RSNA Breast Cancer Detection sur la VM.

Usage (depuis la VM, dans le venv) :
    ~/breast_cancer/.venv/bin/python ~/breast_cancer/download_rsna.py

Le telechargement dure plusieurs heures -> lancer en tmux pour survivre aux
coupures SSH :
    tmux new-session -d -s dl \\
      "~/breast_cancer/.venv/bin/python ~/breast_cancer/download_rsna.py \\
       2>&1 | tee ~/breast_cancer/data/rsna/download.log"

Suivi :
    tail -f ~/breast_cancer/data/rsna/download.log
    tmux attach -t dl      # Ctrl-b puis d pour detacher
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

COMPETITION = "rsna-breast-cancer-detection"
ZIP_NAME = f"{COMPETITION}.zip"
# Il faut de la place pour le zip (~314 Go) ET son extraction (~314 Go) :
# pic ~628 Go. On verifie une marge de securite.
MIN_FREE_GB = 630


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RSNA breast cancer dataset")
    parser.add_argument(
        "--dest",
        default=os.path.expanduser("~/breast_cancer/data/rsna"),
        help="dossier de destination",
    )
    parser.add_argument(
        "--no-unzip", action="store_true", help="ne pas extraire le zip apres download"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="conserver le zip apres extraction"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-telecharger meme si le zip existe deja"
    )
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    free = free_gb(dest)
    print(f"[info] Espace libre sur {dest}: {free:.0f} Go", flush=True)
    if free < MIN_FREE_GB:
        print(
            f"[ATTENTION] Moins de {MIN_FREE_GB} Go libres. Le pic zip+extraction "
            "(~628 Go) risque de saturer le disque.",
            flush=True,
        )
        print(
            "            Option : relancer avec --no-unzip, puis extraire/supprimer "
            "au fil de l'eau.",
            flush=True,
        )

    # L'import declenche l'authentification (lit ~/.kaggle/kaggle.json)
    import kaggle

    kaggle.api.authenticate()
    print(f"[info] Authentifie. Telechargement de '{COMPETITION}' -> {dest}", flush=True)

    kaggle.api.competition_download_files(
        COMPETITION, path=str(dest), quiet=False, force=args.force
    )

    zip_path = dest / ZIP_NAME
    if not zip_path.exists():
        print(f"[erreur] Zip introuvable apres telechargement : {zip_path}", file=sys.stderr)
        sys.exit(1)
    size_gb = zip_path.stat().st_size / 1024**3
    print(f"[info] Telechargement termine : {zip_path} ({size_gb:.1f} Go)", flush=True)

    if args.no_unzip:
        print("[info] --no-unzip : extraction sautee.", flush=True)
        print(f"[OK] Zip pret dans {dest}", flush=True)
        return

    print(f"[info] Espace libre avant extraction : {free_gb(dest):.0f} Go", flush=True)
    print("[info] Extraction en cours...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, dest)
            if i % 1000 == 0 or i == total:
                print(f"  extrait {i}/{total}", flush=True)
    print("[info] Extraction terminee.", flush=True)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"[info] Zip supprime : {zip_path}", flush=True)

    print(f"[OK] Dataset pret dans {dest}", flush=True)


if __name__ == "__main__":
    main()
