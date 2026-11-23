# runs the alpha ladder for one block (see BLOCKS below) and appends one row per run to results/block_<name>.csv
# usage: python experiments/run_block.py B
# safe to stop and restart, runs that are already in the csv get skipped
import csv
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
from emeans_id.real_data import have, load_real
from emeans_id.synthetic import make_data, zador_init

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"

ALL_M = [2, 4, 8, 16, 32, 64, 128]
BLOCKS = {
    # main sweep
    "B": dict(n=100_000, ms=ALL_M, ks=[256, 1024, 4096], Ts=[2, 5], seeds=5),
    # same as B at low m but with revival off (revival hurts at m=2 gaussian, see revival_lowdim.py)
    "B_norev": dict(n=100_000, ms=[2, 4, 8, 16], ks=[1024], Ts=[2, 5], seeds=5, revive=False),
    # same but at the paper's default batch size (knee), only where alpha_law < 1
    "Bp": dict(n=100_000, ms=[2, 4, 8, 32, 128], ks=[256, 1024, 4096], Ts=[2, 5], seeds=3,
               batch="knee"),
    # mechanism
    "C_uniform": dict(n=100_000, ms=[2, 4, 8, 16], ks=[1024], Ts=[2, 5], seeds=5, dist="uniform"),
    "C_student": dict(n=100_000, ms=[2, 4, 8, 16], ks=[1024], Ts=[2, 5], seeds=5, dist="student"),
    "C_zador": dict(n=100_000, ms=[2, 4, 8, 16], ks=[1024], Ts=[2, 5], seeds=5, zador=True),
    # noise, embedded in 128 dims
    "D": dict(n=100_000, ms=[4, 16], ks=[1024], Ts=[2, 5], seeds=5, D=128,
              sigmas=[0.03, 0.1, 0.3]),
    # bigger n
    "E": dict(n=1_000_000, ms=ALL_M, ks=[4096], Ts=[2, 5], seeds=3),
    # real data (datasets that aren't downloaded yet get skipped)
    "F": dict(datasets=["sift", "fashion", "glove", "deep", "pixels"], ks=[256, 1024],
              Ts=[2, 5], seeds=3),
    # tiny one to check the script works
    "smoke": dict(n=20_000, ms=[2, 32], ks=[256], Ts=[2], seeds=2),
    "smoke_real": dict(datasets=["pixels"], ks=[256], Ts=[2], seeds=1),
}

FIELDS = ["block", "dist", "m", "D", "sigma", "zador", "n", "k", "T", "B", "alpha_law", "j",
          "alpha0", "seed", "train", "val", "test", "init_test", "revived", "dead", "secs"]


def done_keys(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, on_bad_lines="skip")  # a killed run can leave half a line at the end
    # df["T"] not df.T, df.T is the transpose...
    return set(zip(df.dist, df.m, df.sigma, df.k, df["T"], df.j, df.seed))


def start_centroids(cfg, k, seed):
    if cfg["zador"]:
        return zador_init(k, cfg["m"], seed=seed, device=dev)
    if cfg["unique"] is not None:
        # pixels: lots of identical colours, so pick the starting centroids among distinct ones
        u = cfg["unique"]
        g = torch.Generator().manual_seed(seed)
        return u[torch.randperm(len(u), generator=g)[:k].to(dev)]
    return None  # emeans picks random data points


def run_rungs(cfg, data, k, T, B, a_law, js, writer, f, done):
    Xtr, Xva, Xte = data
    seen = set()
    init_mse = {}  # test mse of the starting centroids, same for every alpha (needed for normalized regret)
    for j in js:
        alpha0 = min(a_law * 2 ** (j / 2), 1.0)
        if alpha0 in seen:  # several j's can get clipped to 1
            continue
        seen.add(alpha0)
        for seed in range(cfg["seeds"]):
            key = (cfg["dist"], cfg["m"], cfg["sigma"], k, T, j, seed)
            if key in done:
                continue
            init = start_centroids(cfg, k, seed)
            if seed not in init_mse:
                C0, _ = emeans(Xtr, k, 0, B, alpha0, init=init, seed=seed)  # T=0 just returns the init
                init_mse[seed] = exact_mse(Xte, C0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            C, st = emeans(Xtr, k, T, B, alpha0, init=init, seed=seed,
                           revive_every=5 if cfg["revive"] else 0)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            sub = Xtr[:200_000]
            counts = torch.bincount(nearest(sub, C), minlength=k)
            writer.writerow({
                "block": cfg["block"], "dist": cfg["dist"], "m": cfg["m"], "D": cfg["D"],
                "sigma": cfg["sigma"], "zador": cfg["zador"], "n": cfg["n"], "k": k, "T": T,
                "B": B, "alpha_law": a_law, "j": j, "alpha0": alpha0, "seed": seed,
                "train": exact_mse(sub, C), "val": exact_mse(Xva, C), "test": exact_mse(Xte, C),
                "init_test": init_mse[seed], "revived": st["revived"],
                "dead": int((counts < 10).sum()), "secs": secs,
            })
            f.flush()
            done.add(key)


def best_j(path, cfg, k, T):
    # j with the lowest median val mse for this config
    df = pd.read_csv(path, on_bad_lines="skip")
    sub = df[(df.dist == cfg["dist"]) & (df.m == cfg["m"]) & (df.sigma == cfg["sigma"])
             & (df.k == k) & (df["T"] == T)]
    med = sub.groupby("j").val.median()
    return med.idxmin(), med.index.min(), med.index.max(), sub.alpha0.max()


def run_config(cfg, data, spec, path, writer, f, done):
    n = cfg["n"]
    for k in spec["ks"]:
        for T in spec["Ts"]:
            if spec.get("batch") == "knee":
                B = laws.knee_batch(k, n, T)
            else:
                B = laws.our_batch(k, n, T)
            a_law = laws.alpha_law(k, B, n, T)
            if spec.get("batch") == "knee" and a_law >= 1:
                continue  # law is capped here, nothing to learn about its slope
            run_rungs(cfg, data, k, T, B, a_law, range(-8, 7), writer, f, done)

            # if the best rung is at the edge of the ladder, go 4 rungs further (max twice)
            for _ in range(2):
                jb, jlo, jhi, amax = best_j(path, cfg, k, T)
                if jb == jlo:
                    extra = range(jlo - 4, jlo)
                elif jb == jhi and amax < 1.0:
                    extra = range(jhi + 1, jhi + 5)
                else:
                    break
                run_rungs(cfg, data, k, T, B, a_law, extra, writer, f, done)
            print(f"{cfg['block']}: {cfg['dist']} m={cfg['m']} sigma={cfg['sigma']} k={k} T={T} done",
                  flush=True)


def main(name):
    spec = BLOCKS[name]
    path = os.path.join(ROOT, "results", f"block_{name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = done_keys(path)
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    base = {"block": name, "sigma": 0.0, "zador": spec.get("zador", False), "seeds": spec["seeds"],
            "revive": spec.get("revive", True), "unique": None}

    if "datasets" in spec:
        for ds in spec["datasets"]:
            if not have(ds):
                print(f"{ds} not downloaded yet, skipping", flush=True)
                continue
            data = load_real(ds, device=dev)
            cfg = {**base, "dist": ds, "m": 0, "D": data[0].shape[1], "n": len(data[0])}
            if ds == "pixels":
                cfg["unique"] = torch.unique(data[0], dim=0)
            run_config(cfg, data, spec, path, writer, f, done)
            del data
            cfg["unique"] = None
            torch.cuda.empty_cache()
    else:
        n = spec["n"]
        for sigma in spec.get("sigmas", [0.0]):
            for m in spec["ms"]:
                cfg = {**base, "dist": spec.get("dist", "gaussian"), "m": m, "D": spec.get("D", m),
                       "sigma": sigma, "n": n}
                data = make_data(n, m, dist=cfg["dist"], D=spec.get("D"), sigma=sigma, seed=0,
                                 device=dev)
                run_config(cfg, data, spec, path, writer, f, done)
                del data
                torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main(sys.argv[1])
