#!/usr/bin/env python
"""Sweep de configs (batch/workers/pin/prefetch/normalisation-GPU) pour le
finetuning GMIC. Chaque config tourne dans un sous-process isole sous un scope
systemd avec plafond memoire (la VM ne peut pas tomber). On mesure debit (img/s),
pic VRAM et pic RAM (cgroup). Sortie : table triee + JSON.
"""
import argparse, json, os, subprocess, sys, time

REPO = "/home/joshua/breast_cancer"
PY = REPO + "/.venv/bin/python"
os.chdir(REPO)
sys.path.insert(0, REPO + "/GMIC")
sys.path.insert(0, REPO)


def cgroup_peak_gb():
    try:
        path = open("/proc/self/cgroup").read().strip().split("::")[-1]
        for f in ("memory.peak", "memory.current"):
            fp = "/sys/fs/cgroup" + path + "/" + f
            if os.path.exists(fp):
                return int(open(fp).read()) / 1e9
    except Exception:
        pass
    return float("nan")


def run_one(cfg):
    import numpy as np, torch, torch.nn as nn
    from torch.amp import autocast, GradScaler
    from torch.utils.data import DataLoader
    from fine_tuning.dataset_gmic import (load_and_split3, build_samples,
                                          GMICViewDataset, _make_sampler, INPUT_SIZE)
    from fine_tuning.gmic_finetune import build_gmic, default_parameters, param_groups
    from fine_tuning.train_gmic import compute_loss

    device = torch.device("cuda")
    tr_ex, _, _, _ = load_and_split3("data/preprocess_image/data.pkl")
    samples = build_samples(tr_ex, "data/preprocess_image/cropped_images")
    ds = GMICViewDataset(samples, augment=True, input_size=INPUT_SIZE,
                         to_uint8=bool(cfg["gpu_norm"]))
    kw = dict(num_workers=cfg["workers"], pin_memory=bool(cfg["pin"]))
    if cfg["workers"] > 0:
        kw.update(prefetch_factor=cfg["prefetch"], persistent_workers=False)
    loader = DataLoader(ds, batch_size=cfg["batch"], sampler=_make_sampler(samples),
                        drop_last=True, **kw)
    model = build_gmic("GMIC/models/sample_model_1.p",
                       default_parameters(percent_t=0.02, gpu_number=0), device)
    model.train()
    opt = torch.optim.Adam(param_groups(model, 1e-5, 1e-4), weight_decay=1e-5)
    scaler = GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()

    it = iter(loader); done = 0; t0 = None
    target = cfg["warmup"] + cfg["steps"]
    while done < target:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x = x.to(device, non_blocking=True); yt = y.to(device)
        if cfg["gpu_norm"]:
            x = x.float()
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            s = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
            x = (x - m) / s
        opt.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            yf, yg, yl, sal = model(x)
        total, _ = compute_loss(yf, yg, yl, sal, yt, 1e-4)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        done += 1
        if done == cfg["warmup"]:
            torch.cuda.synchronize(); t0 = time.time()
    torch.cuda.synchronize()
    el = time.time() - t0
    sps = cfg["steps"] / el
    vram = torch.cuda.max_memory_allocated() / 1e9
    ram = cgroup_peak_gb()
    print("RESULT batch=%d workers=%d pin=%d prefetch=%d gpu_norm=%d "
          "steps_s=%.3f imgs_s=%.1f vram_gb=%.2f ram_gb=%.2f" %
          (cfg["batch"], cfg["workers"], cfg["pin"], cfg["prefetch"], cfg["gpu_norm"],
           sps, sps * cfg["batch"], vram, ram), flush=True)


def launch(cfg, mem_max="42G", timeout=300):
    args = ["systemd-run", "--user", "--scope", "--quiet",
            "-p", "MemoryMax=" + mem_max, "-p", "MemorySwapMax=0", "--",
            PY, REPO + "/_gmic_sweep.py", "--run-one",
            "--batch", str(cfg["batch"]), "--workers", str(cfg["workers"]),
            "--pin", str(cfg["pin"]), "--prefetch", str(cfg["prefetch"]),
            "--gpu-norm", str(cfg["gpu_norm"]),
            "--warmup", str(cfg["warmup"]), "--steps", str(cfg["steps"])]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(cfg, status="TIMEOUT")
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        st = "OOM" if p.returncode in (137, -9, 9) else "FAIL(%s)" % p.returncode
        tail = (p.stderr.strip().splitlines()[-2:] if p.stderr.strip() else [])
        return dict(cfg, status=st, err=" | ".join(tail))
    d = dict(tok.split("=") for tok in line.split()[1:])
    return dict(cfg, status="OK", steps_s=float(d["steps_s"]), imgs_s=float(d["imgs_s"]),
                vram_gb=float(d["vram_gb"]), ram_gb=float(d["ram_gb"]))


def base(b, w, pin, pf, gn, wu, st):
    return dict(batch=b, workers=w, pin=pin, prefetch=pf, gpu_norm=gn, warmup=wu, steps=st)


def fmt(r):
    if r["status"] != "OK":
        return ("  b=%-2d w=%d pin=%d pf=%d gn=%d  -> %s %s"
                % (r["batch"], r["workers"], r["pin"], r["prefetch"], r["gpu_norm"],
                   r["status"], r.get("err", "")))
    return ("  b=%-2d w=%d pin=%d pf=%d gn=%d  | %6.1f img/s | %5.2f st/s | VRAM %4.1f GB | RAM %4.1f GB"
            % (r["batch"], r["workers"], r["pin"], r["prefetch"], r["gpu_norm"],
               r["imgs_s"], r["steps_s"], r["vram_gb"], r["ram_gb"]))


def orchestrate():
    WU, ST = 8, 25
    results = []

    print("=== PHASE 1 : data-loading (batch=16, gpu_norm=0) ===", flush=True)
    p1 = [base(16, w, pin, pf, 0, WU, ST) for (w, pin, pf) in
          [(2, 0, 1), (2, 1, 2), (4, 0, 1), (4, 1, 2), (6, 1, 2), (8, 1, 2), (8, 0, 2)]]
    for c in p1:
        r = launch(c); results.append(r); print(fmt(r), flush=True)

    ok1 = [r for r in results if r["status"] == "OK" and r["ram_gb"] < 38]
    if ok1:
        bestdl = max(ok1, key=lambda r: r["imgs_s"])
        bw, bpin, bpf = bestdl["workers"], bestdl["pin"], bestdl["prefetch"]
    else:
        bw, bpin, bpf = 4, 0, 1
    print("\n-> meilleur data-loading : workers=%d pin=%d prefetch=%d\n" % (bw, bpin, bpf), flush=True)

    print("=== PHASE 2 : batch x normalisation-GPU (workers=%d pin=%d pf=%d) ===" % (bw, bpin, bpf), flush=True)
    p2 = [base(b, bw, bpin, bpf, gn, WU, ST) for b in (8, 16, 32, 48, 64) for gn in (0, 1)]
    for c in p2:
        r = launch(c); results.append(r); print(fmt(r), flush=True)

    ok = [r for r in results if r["status"] == "OK"]
    ok.sort(key=lambda r: r["imgs_s"], reverse=True)
    print("\n" + "=" * 70)
    print("CLASSEMENT (par debit, configs OK) :")
    for r in ok:
        print(fmt(r))
    bad = [r for r in results if r["status"] != "OK"]
    if bad:
        print("\nConfigs ecartees :")
        for r in bad:
            print(fmt(r))
    if ok:
        safe = [r for r in ok if r["ram_gb"] < 30]
        win = (safe or ok)[0]
        print("\n>>> RECOMMANDATION (rapide + RAM < 30 GB) :")
        print(fmt(win))
    json.dump(results, open(REPO + "/_gmic_sweep_results.json", "w"), indent=2)
    print("\nJSON -> _gmic_sweep_results.json")
    print("=" * 70, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pin", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--gpu-norm", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()
    if a.run_one:
        run_one(dict(batch=a.batch, workers=a.workers, pin=a.pin, prefetch=a.prefetch,
                     gpu_norm=a.gpu_norm, warmup=a.warmup, steps=a.steps))
    else:
        orchestrate()


if __name__ == "__main__":
    main()
