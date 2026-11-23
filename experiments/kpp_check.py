# does greedy k-means++ already fix the low-m problem?
# compares starting points for lloyd and e-means at m = 2, 4, 8
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
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data, zador_init

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
n, T = 100_000, 5

rows = []
for m in [2, 4, 8]:
    Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
    for k in [256, 1024]:
        B = laws.our_batch(k, n, T)
        a = laws.alpha_law(k, B, n, T)
        for seed in range(3):
            t0 = time.perf_counter()
            kpp = kmeanspp(Xtr, k, seed=seed)
            torch.cuda.synchronize()
            kpp_secs = time.perf_counter() - t0
            zad = zador_init(k, m, seed=seed, device=dev)
            res = {
                "lloyd, random": lloyd(Xtr, k, 25, seed=seed)[0],
                "lloyd, k-means++": lloyd(Xtr, k, 25, init=kpp, seed=seed)[0],
                "lloyd, zador": lloyd(Xtr, k, 25, init=zad, seed=seed)[0],
                "emeans, random": emeans(Xtr, k, T, B, a, seed=seed)[0],
                "emeans, k-means++": emeans(Xtr, k, T, B, a, init=kpp, seed=seed)[0],
                "emeans, zador": emeans(Xtr, k, T, B, a, init=zad, seed=seed)[0],
                "emeans a=1 no revival, random": emeans(Xtr, k, T, B, 1.0, revive_every=0, seed=seed)[0],
                "k-means++ alone": kpp,
            }
            for name, C in res.items():
                rows.append({"m": m, "k": k, "seed": seed, "method": name,
                             "test": exact_mse(Xte, C), "kpp_secs": kpp_secs})
        print(f"m={m} k={k} done", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "kpp_check.csv"), index=False)

# % above the best method for each (m, k)
g = df.groupby(["m", "k", "method"]).test.mean().reset_index()
g["vs_best_%"] = g.groupby(["m", "k"]).test.transform(lambda s: 100 * (s / s.min() - 1))
pd.set_option("display.width", 200)
print(g.pivot_table(index="method", columns=["m", "k"], values="vs_best_%").round(1).to_string())
print("\nzador high-rate bound at m=2 (k=1024): 0.001968, k=256: 0.007873")
print(g[g.m == 2].pivot_table(index="method", columns="k", values="test").round(6).to_string())
print(f"\nk-means++ init time (mean): {df.kpp_secs.mean():.2f}s")
