# claim (b): e-means seems to throw away a good k-means++ start at low m. is revival the reason?
# same k-means++ start, e-means with revival on/off and alpha0 = law or 1, vs lloyd from that start
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.init import kmeanspp
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
n, T = 100_000, 5

rows = []
for m in [2, 3, 4]:
    Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
    for k in [256, 1024]:
        B = laws.our_batch(k, n, T)
        a = laws.alpha_law(k, B, n, T)
        for seed in range(5):
            kpp = kmeanspp(Xtr, k, seed=seed)
            res = {
                "kpp alone": kpp,
                "lloyd25": lloyd(Xtr, k, 25, init=kpp, seed=seed)[0],
                "emeans law, revival on": emeans(Xtr, k, T, B, a, init=kpp, seed=seed)[0],
                "emeans law, revival off": emeans(Xtr, k, T, B, a, init=kpp, seed=seed, revive_every=0)[0],
                "emeans a=1, revival on": emeans(Xtr, k, T, B, 1.0, init=kpp, seed=seed)[0],
                "emeans a=1, revival off": emeans(Xtr, k, T, B, 1.0, init=kpp, seed=seed, revive_every=0)[0],
            }
            for name, C in res.items():
                rows.append({"m": m, "k": k, "seed": seed, "method": name, "test": exact_mse(Xte, C)})
        print(f"m={m} k={k} done", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "revival_kpp.csv"), index=False)
pd.set_option("display.width", 200)

# % vs lloyd from the same k-means++ start (mean over seeds)
g = df.groupby(["m", "k", "method"]).test.mean().unstack("method")
rel = 100 * (g.div(g["lloyd25"], axis=0) - 1)
print("\n% test mse vs lloyd@25 from the same k-means++ start (5 seeds)")
print(rel.round(2).to_string())

# paired: revival off vs on at the law's alpha
p = df.pivot_table(index=["m", "k", "seed"], columns="method", values="test")
d = 100 * (p["emeans law, revival off"] / p["emeans law, revival on"] - 1)
s = d.groupby(["m", "k"]).agg(["mean", "std", lambda x: int((x < 0).sum())])
s.columns = ["off_vs_on_%", "sd", "seeds_off_better"]
s["se"] = s.sd / 5 ** 0.5
print("\npaired, revival off vs on (law alpha, k-means++ start), negative = off better")
print(s.round(3).to_string())
