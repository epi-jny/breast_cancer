import sys
import kaggle

COMP = "rsna-breast-cancer-detection"
try:
    kaggle.api.authenticate()
    print("[diag] authenticate() OK")
except Exception as e:
    print(f"[diag] authenticate() FAIL: {type(e).__name__}: {e}")
    sys.exit(2)

try:
    files = kaggle.api.competition_list_files(COMP)
    files = list(files)
    print(f"[diag] competition_list_files OK -> {len(files)} fichier(s)")
    for f in files[:10]:
        print("   -", f)
except Exception as e:
    print(f"[diag] competition_list_files FAIL: {type(e).__name__}: {e}")
    sys.exit(3)

print(f"[diag] kaggle version: {getattr(kaggle, '__version__', 'inconnue')}")
