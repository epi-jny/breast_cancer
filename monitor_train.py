#!/usr/bin/env python
"""Dashboard live : CPU / RAM / GPU + avancement de l'entrainement GMIC.

Sans dependance externe : lit /proc (CPU, RAM) + nvidia-smi (GPU) et parse le
logs.json du run pour l'avancement.

Usage (sur la VM) :
    ./.venv/bin/python monitor_train.py                       # run par defaut
    ./.venv/bin/python monitor_train.py <run_dir> -i 1        # autre run, refresh 1s
    ./.venv/bin/python monitor_train.py --once                # un seul snapshot (pipe-friendly)

Ctrl-C pour quitter.
"""
import argparse
import json
import os
import subprocess
import time

DEFAULT_RUN = "fine_tuning/checkpoints/gmic_ft_exam"


def _f(x, default=0.0):
    try:
        return float(str(x).strip())
    except (ValueError, TypeError):
        return default


def read_cpu_times():
    with open("/proc/stat") as f:
        parts = [float(x) for x in f.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)  # idle + iowait
    return idle, sum(parts)


def cpu_percent(prev, cur):
    d_idle = cur[0] - prev[0]
    d_total = cur[1] - prev[1]
    return 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else 0.0


def mem_info():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            info[k] = int(v.strip().split()[0])  # kB
    total = info["MemTotal"] / 1024 / 1024  # Go
    avail = info.get("MemAvailable", info["MemFree"]) / 1024 / 1024
    return total - avail, total


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        v = [x.strip() for x in out.strip().split("\n")[0].split(",")]
        return {"util": _f(v[0]), "mem_used": _f(v[1]), "mem_total": _f(v[2]),
                "temp": _f(v[3]), "power": _f(v[4]), "power_max": _f(v[5], 1.0)}
    except Exception:
        return None


def bar(pct, width=30):
    pct = max(0.0, min(100.0, pct))
    fill = int(round(width * pct / 100.0))
    return "[" + "#" * fill + "." * (width - fill) + f"] {pct:5.1f}%"


def fmt_dur(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def training_alive(run_dir):
    """True si un process train_gmic est vivant.

    Le run dir horodate est genere DANS le script (absent de l'argv) -> on ne
    peut pas toujours matcher le chemin. Match exact si possible, sinon
    n'importe quel process train_gmic (machine mono-run en pratique).
    """
    try:
        out = subprocess.check_output(["pgrep", "-af", "train_gmic"],
                                      text=True, timeout=5)
    except Exception:
        return False
    lines = [l for l in out.splitlines() if "train_gmic" in l]
    base = os.path.basename(run_dir.rstrip("/"))
    if any(base in l or run_dir.rstrip("/") in l for l in lines):
        return True
    return bool(lines)


def load_logs(run_dir):
    p = os.path.join(run_dir, "logs.json")
    if not os.path.exists(p):
        return None, None
    try:
        return json.load(open(p)), os.path.getmtime(p)
    except Exception:
        return None, None


def render(run_dir, cpu, mem, la, g, logs, mtime, epochs_cap, patience, min_delta, alive):
    lines = []
    lines.append("=" * 66)
    status = "EN COURS" if alive else "ARRETE/TERMINE"
    run_name = os.path.basename(run_dir.rstrip("/"))
    lines.append(f" MONITOR  {time.strftime('%H:%M:%S')}   run={run_name}   [{status}]")
    lines.append("=" * 66)
    used, total = mem
    lines.append(f" CPU  {bar(cpu)}")
    lines.append(f"      load {la[0]}/{la[1]}/{la[2]} sur {os.cpu_count()} vCPU")
    lines.append(f" RAM  {bar(100 * used / total)}  {used:5.1f}/{total:4.0f} Go")
    if g:
        lines.append(f" GPU  {bar(g['util'])}  {g['temp']:.0f}C  "
                     f"{g['power']:.0f}/{g['power_max']:.0f}W")
        lines.append(f" VRAM {bar(100 * g['mem_used'] / max(g['mem_total'], 1))}  "
                     f"{g['mem_used']/1024:4.1f}/{g['mem_total']/1024:4.1f} Go")
    else:
        lines.append(" GPU  (nvidia-smi indisponible)")
    lines.append("-" * 66)
    if logs:
        done = len(logs)
        times = [e.get("time_s") for e in logs if e.get("time_s")]
        avg_t = sum(times) / len(times) if times else 0.0
        last = logs[-1]
        best, best_ep, since = -1.0, None, 0
        for e in logs:
            a = e.get("exam_auc_malignant")
            if a is not None and a == a and a > best + min_delta:
                best, best_ep, since = a, e["epoch"], 0
            else:
                since += 1

        def fnum(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "  nan"

        lines.append(f" Epochs terminees : {done}/{epochs_cap}   (~{avg_t:.0f}s/epoch)")
        lines.append(f" Derniere ep {last['epoch']:>2}: "
                     f"train_loss={fnum(last.get('train_loss'))}  "
                     f"val_loss={fnum(last.get('val_loss'))}")
        lines.append(f"   EXAM AUC_mal={fnum(last.get('exam_auc_malignant'))}  "
                     f"(vue={fnum(last.get('auc_malignant'))}  "
                     f"ben_exam={fnum(last.get('exam_auc_benign'))})")
        lines.append(f" BEST EXAM AUC_mal={fnum(best)} @ep {best_ep}  |  "
                     f"early-stop {since}/{patience}")
        if alive:
            remaining = max(0, epochs_cap - done)
            lines.append(f" ETA plafond ({remaining} ep restantes) ~ {fmt_dur(remaining * avg_t)}"
                         "   (early stop possible avant)")
            if avg_t > 0 and mtime:
                since_log = time.time() - mtime
                pct = min(100.0, 100.0 * since_log / avg_t)
                lines.append(f" Epoch en cours ~{bar(pct)}  ({since_log:.0f}s ecoulees)")
        else:
            stop = "early stopping" if since >= patience else "fin"
            lines.append(f" >> ENTRAINEMENT TERMINE ({stop}). best.pt = ep {best_ep} "
                         f"(EXAM AUC_mal={fnum(best)})")
    else:
        lines.append(" (pas encore de logs.json -- epoch 0 en cours, ~12 min)")
    lines.append("=" * 66)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN)
    ap.add_argument("-i", "--interval", type=float, default=2.0)
    ap.add_argument("--epochs-cap", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--once", action="store_true", help="un seul snapshot puis quitte")
    args = ap.parse_args()

    prev = read_cpu_times()
    time.sleep(0.3 if args.once else args.interval)
    while True:
        cur = read_cpu_times()
        cpu = cpu_percent(prev, cur)
        prev = cur
        logs, mtime = load_logs(args.run_dir)
        screen = render(args.run_dir, cpu, mem_info(), loadavg(), gpu_info(),
                        logs, mtime, args.epochs_cap, args.patience, args.min_delta,
                        training_alive(args.run_dir))
        if args.once:
            print(screen)
            return
        os.system("clear")
        print(screen)
        print(" Ctrl-C pour quitter")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
