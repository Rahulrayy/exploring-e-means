# proper check of the revival threshold tau = 0.01 (vs the paper's 0.1 and no revival)
# appends one row per run to results/tau_confirm.csv, safe to stop and restart
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.init import kmeanspp
from emeans_id.real_data import load_real
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
OUT = os.path.join(ROOT, "results", "tau_confirm.csv")
FIELDS = ["data", "m", "n", "k", "T", "start", "tau", "seed", "B", "alpha0", "val", "test",
          "revived", "secs"]
TAUS = [0.1, 0.01, 0.0]  # 0 = revival off
KS = [256, 1024, 4096]
TS = [2, 5]
SEEDS = range(5)
DATASETS = [("gaussian", m) for m in [2, 3, 4, 8, 16, 128]] + \
           [(d, 0) for d in ["pixels", "fashion", "sift", "deep", "glove"]]


def done_keys():
    if not os.path.exists(OUT):
        return set()
    df = pd.read_csv(OUT, on_bad_lines="skip")
    return set(zip(df.data, df.m, df.k, df["T"], df.start, df.tau, df.seed))


def main():
    done = done_keys()
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    for data, m in DATASETS:
        todo = [(data, m, k, T, s, tau, seed) for k in KS for T in TS for s in ["random", "kpp"]
                for tau in TAUS for seed in SEEDS]
        if all(key in done for key in todo):
            continue
        if data == "gaussian":
            Xtr, Xva, Xte = make_data(100_000, m, seed=0, device=dev)
        else:
            Xtr, Xva, Xte = load_real(data, device=dev)
        n = len(Xtr)
        for k in KS:
            if n / k < 8:
                continue
            for seed in SEEDS:
                keys_here = [(data, m, k, T, s, tau, seed) for T in TS for s in ["random", "kpp"] for tau in TAUS]
                if all(key in done for key in keys_here):
                    continue
                kpp = kmeanspp(Xtr, k, seed=seed)  # one k-means++ start per (data, k, seed), reused for every T and tau
                for T in TS:
                    B = laws.our_batch(k, n, T)
                    a = laws.alpha_law(k, B, n, T)
                    for start in ["random", "kpp"]:
                        init = kpp if start == "kpp" else None
                        for tau in TAUS:
                            key = (data, m, k, T, start, tau, seed)
                            if key in done:
                                continue
                            kw = {"revive_every": 0} if tau == 0 else {"tau": tau}
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            C, st = emeans(Xtr, k, T, B, a, init=init, seed=seed, **kw)
                            torch.cuda.synchronize()
                            secs = time.perf_counter() - t0
                            w.writerow({"data": data, "m": m, "n": n, "k": k, "T": T, "start": start,
                                        "tau": tau, "seed": seed, "B": B, "alpha0": a,
                                        "val": exact_mse(Xva, C), "test": exact_mse(Xte, C),
                                        "revived": st["revived"], "secs": secs})
                            f.flush()
                            done.add(key)
            print(f"{data} m={m} k={k} done", flush=True)
        del Xtr, Xva, Xte
        torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()
