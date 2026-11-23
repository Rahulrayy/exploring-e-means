# sanity checks to run before the main sweep (S4b, S5, S6, S7)
# usage: python experiments/sanity_checks.py s5      (or s4b / s6 / s7 / all)
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse, nearest
from emeans_id.emeans import emeans
from emeans_id.lloyd import lloyd
from emeans_id.real_data import load_ann
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
pd.set_option("display.width", 200)


def timed(fn, *args, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def free_gpu():
    torch.cuda.empty_cache()


def s4b():
    # rotation invariance again but in fp32 with revival on.
    # runs won't match one by one (k-means is chaotic) so compare over 10 seeds
    m, D, n, k, T = 8, 128, 100_000, 1024, 2
    B = laws.our_batch(k, n, T)
    a = laws.alpha_law(k, B, n, T)
    Z, _, Zte = make_data(n, m, seed=0, device=dev)
    X, _, Xte = make_data(n, m, D=D, seed=0, device=dev)
    rows = []
    for seed in range(10):
        Cz, _ = emeans(Z, k, T, B, a, seed=seed)
        Cx, _ = emeans(X, k, T, B, a, seed=seed)
        rows.append({"seed": seed, "mse_z": exact_mse(Zte, Cz), "mse_x": exact_mse(Xte, Cx)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "s4b_rotation.csv"), index=False)
    diff = df.mse_x.mean() - df.mse_z.mean()
    sd = df.mse_z.std()
    print(df.to_string(index=False))
    print(f"\nS4b: mean diff {diff:.2e}, seed std {sd:.2e} -> {'PASS' if abs(diff) < sd else 'FAIL'}")
    del Z, X, Zte, Xte
    free_gpu()


def s5():
    # mini reproduction on sift: e-means with the paper's own laws vs our lloyd@25
    rows = []
    for n in [100_000, 1_000_000]:
        Xtr, Xva, Xte = load_ann("sift", n_train=n, device=dev)
        for k in [1024, 4096]:
            for seed in range(3):
                (C, _), secs = timed(lloyd, Xtr, k, iters=25, seed=seed)
                rows.append({"n": n, "k": k, "method": "lloyd", "T": 25, "seed": seed,
                             "B": n, "alpha0": None, "secs": secs,
                             "train": exact_mse(Xtr[:200_000], C), "test": exact_mse(Xte, C)})
                for T in [1, 2, 3, 4, 5]:
                    B = laws.knee_batch(k, n, T)
                    a = laws.alpha_law(k, B, n, T)
                    (C, st), secs = timed(emeans, Xtr, k, T, B, a, seed=seed)
                    rows.append({"n": n, "k": k, "method": "emeans", "T": T, "seed": seed,
                                 "B": B, "alpha0": a, "secs": secs,
                                 "train": exact_mse(Xtr[:200_000], C), "test": exact_mse(Xte, C)})
            print(f"done n={n} k={k}")
        del Xtr, Xva, Xte
        free_gpu()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "s5_sift.csv"), index=False)

    # % difference to lloyd@25 (mean over seeds), negative = e-means better
    g = df.groupby(["n", "k", "method", "T"]).agg(test=("test", "mean"), train=("train", "mean"),
                                                  secs=("secs", "mean"), B=("B", "first"),
                                                  alpha0=("alpha0", "first")).reset_index()
    ref = g[g.method == "lloyd"].set_index(["n", "k"])
    g["test_vs_lloyd_%"] = [100 * (r.test / ref.loc[(r.n, r.k), "test"] - 1) for r in g.itertuples()]
    g["train_vs_lloyd_%"] = [100 * (r.train / ref.loc[(r.n, r.k), "train"] - 1) for r in g.itertuples()]
    print(g.round(4).to_string(index=False))

    print("\nS5: first T where e-means beats lloyd@25 on test (mean over seeds)")
    for (n, k), sub in g[g.method == "emeans"].groupby(["n", "k"]):
        win = sub[sub["test_vs_lloyd_%"] < 0]
        first = int(win["T"].min()) if len(win) else None
        print(f"  n={n:>8} k={k:>5}: T={first}  -> {'PASS' if first is not None and first <= 5 else 'FAIL'}")


S6_VARIANTS = {"grace": {}, "no_grace": {"grace": False}, "off": {"revive_every": 0}}


def s6_run(rows, tag, Xtr, Xte, k, T, B, a, m):
    # revival ablation: grace period (our default), no grace (paper read literally), no revival
    for name, kw in S6_VARIANTS.items():
        for seed in range(3):
            C, st = emeans(Xtr, k, T, B, a, seed=seed, **kw)
            counts = torch.bincount(nearest(Xtr, C), minlength=k)
            rows.append({"data": tag, "m": m, "k": k, "T": T, "B": B, "alpha0": a, "variant": name,
                         "seed": seed, "revived": st["revived"],
                         "dead": int((counts < 10).sum()), "test": exact_mse(Xte, C)})


def s6_synth():
    # synthetic gaussian with the batch rule we'll use in block B
    rows = []
    n, k = 100_000, 1024
    for m in [2, 16, 128]:
        Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
        for T in [2, 5]:
            B = laws.our_batch(k, n, T)
            s6_run(rows, f"gauss_m{m}", Xtr, Xte, k, T, B, laws.alpha_law(k, B, n, T), m)
        del Xtr, Xte
        free_gpu()
    s6_report(rows, "s6_revival_synth.csv")


def s6_sift():
    # sift with the paper's own laws
    rows = []
    for n, k in [(100_000, 1024), (1_000_000, 4096)]:
        Xtr, _, Xte = load_ann("sift", n_train=n, device=dev)
        T = 3
        B = laws.knee_batch(k, n, T)
        s6_run(rows, f"sift{n // 1000}k", Xtr, Xte, k, T, B, laws.alpha_law(k, B, n, T), None)
        del Xtr, Xte
        free_gpu()
    s6_report(rows, "s6_revival_sift.csv")


def s6():
    s6_synth()
    s6_sift()


def s6_report(rows, fname):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, fname), index=False)
    g = df.groupby(["data", "T", "variant"]).agg(test=("test", "mean"), test_sd=("test", "std"),
                                                 revived=("revived", "mean"),
                                                 dead=("dead", "mean")).reset_index()
    ref = g[g.variant == "grace"].set_index(["data", "T"])["test"]
    g["vs_grace_%"] = [100 * (r.test / ref.loc[(r.data, r.T)] - 1) for r in g.itertuples()]
    print(g.round(5).to_string(index=False))


def s7():
    # timing for the heaviest-looking config of each block (training and eval separately)
    def bench(block, Xtr, Xva, Xte, k, T, B):
        a = laws.alpha_law(k, B, len(Xtr), T)
        emeans(Xtr, k, T, B, a, seed=0)  # warmup
        (C, st), secs = timed(emeans, Xtr, k, T, B, a, seed=1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        exact_mse(Xtr[:200_000], C), exact_mse(Xva, C), exact_mse(Xte, C)
        torch.cuda.synchronize()
        ev = time.perf_counter() - t0
        out = {"block": block, "n": len(Xtr), "d": Xtr.shape[1], "k": k, "T": T, "B": B,
               "updates": st["updates"], "train_secs": secs, "eval_secs": ev}
        print(out)
        return out

    rows = []
    n = 100_000
    Xtr, Xva, Xte = make_data(n, 128, seed=0, device=dev)
    for k in [256, 1024, 4096]:
        rows.append(bench("B", Xtr, Xva, Xte, k, 5, laws.our_batch(k, n, 5)))
    rows.append(bench("B'", Xtr, Xva, Xte, 4096, 5, laws.knee_batch(4096, n, 5)))
    del Xtr, Xva, Xte

    Xtr, Xva, Xte = make_data(n, 16, D=128, sigma=0.1, seed=0, device=dev)
    rows.append(bench("D", Xtr, Xva, Xte, 1024, 5, laws.our_batch(1024, n, 5)))
    del Xtr, Xva, Xte
    free_gpu()

    n = 1_000_000
    Xtr, Xva, Xte = make_data(n, 128, seed=0, device=dev)
    rows.append(bench("E", Xtr, Xva, Xte, 4096, 5, laws.our_batch(4096, n, 5)))
    del Xtr, Xva, Xte
    free_gpu()

    Xtr, Xva, Xte = load_ann("sift", device=dev)
    rows.append(bench("F", Xtr, Xva, Xte, 1024, 5, laws.our_batch(1024, len(Xtr), 5)))
    del Xtr, Xva, Xte
    free_gpu()

    df = pd.DataFrame(rows)
    df["total_secs"] = df.train_secs + df.eval_secs
    df.to_csv(os.path.join(RES, "s7_timing.csv"), index=False)
    print(df.round(3).to_string(index=False))

    # rough upper bound: every run in a block as slow as its slowest case
    counts = {"B": 3150, "B'": 1125, "C": 1800, "D": 900, "E": 630, "F": 900, "G": 693}
    worst = df.groupby("block").total_secs.max().to_dict()
    worst["C"] = worst["B"]
    worst["G"] = worst["B"]
    hours = sum(counts[b] * worst[b] for b in counts) / 3600
    print(f"\nS7: worst-case estimate for all blocks ~ {hours:.1f} GPU-hours")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if "all" in todo:
        todo = ["s4b", "s5", "s6", "s7"]
    for name in todo:
        print(f"\n===== {name} =====")
        globals()[name]()
