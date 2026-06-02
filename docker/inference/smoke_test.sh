#!/usr/bin/env bash
# Smoke-test de l'image GMIC : vérifie que la chaîne complète tourne de bout en
# bout SUR LE SAMPLE EMBARQUÉ. Conçu pour être lancé DANS le conteneur :
#   docker run --rm gmic-inference:cpu smoke
# (un seul conteneur => l'état preprocess persiste entre les étapes).
#
# Code retour 0 = tout passe ; != 0 = échec (utilisable en CI / make docker-test).
set -euo pipefail

# Sortie sous /app (PROJECT_ROOT) : eval_gmic.py calcule des chemins relatifs
# à la racine projet. Conteneur éphémère (--rm) → écriture sans conséquence.
OUT=/app/data/preprocess_image/smoke
rm -rf "$OUT"; mkdir -p "$OUT"

fail() { echo "❌ SMOKE FAIL — $1" >&2; exit 1; }

echo "=============================================================="
echo " SMOKE TEST — image GMIC tout-en-un"
echo "=============================================================="

echo
echo "## 1/4 — Tests unitaires (pytest validate_input)"
python -m pytest scripts/test_validate_input.py -q || fail "tests unitaires"

echo
echo "## 2/4 — Preprocessing du sample (crop + resize)"
python scripts/preprocess.py --input-dir data/raw/sample \
    --output-dir "$OUT" --num-processes 2
[ -f "$OUT/data.pkl" ] || fail "data.pkl non généré par le preprocessing"

echo
echo "## 3/4 — Inférence GMIC"
python scripts/inference.py --output-dir "$OUT" --model-index 1
[ -f "$OUT/predictions.csv" ] || fail "predictions.csv non généré"
rows=$(($(wc -l < "$OUT/predictions.csv") - 1))
[ "$rows" -eq 8 ] || fail "attendu 8 prédictions, obtenu $rows"

echo
echo "## 4/4 — Évaluation des prédictions"
python scripts/eval_gmic.py --predictions "$OUT/predictions.csv" || fail "évaluation"

echo
echo "=============================================================="
echo " ✅ SMOKE OK — pytest + preprocess + inférence ($rows prédictions) + éval"
echo "=============================================================="
