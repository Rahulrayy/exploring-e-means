# follow-up to S6: revival seemed to *hurt* at m=2. check with 10 paired seeds,
# and look at how many clusters a converged lloyd solution has below the revival
# threshold (0.1 * mean count). if that's a lot, revival is fighting the optimum
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse, nearest
from emeans_id.emeans import emeans
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
n, k = 100_000, 1024

rows = []
small = []
for dist in ["gaussian", "uniform"]:
    for m in [2, 4, 8, 16]:
        Xtr, _, Xte = make_data(n, m, dist=dist, seed=0, device=dev)

        # converged-ish lloyd: what fraction of clusters is below 0.1 * mean count?
        C, _ = lloyd(Xtr, k, iters=100, seed=0)
        counts = torch.bincount(nearest(Xtr, C), minlength=k).float()
        frac = (counts < 0.1 * counts.mean()).float().mean().item()
        small.append({"dist": dist, "m": m, "below_threshold_%": 100 * frac,
                      "min_count": counts.min().item(), "mean_count": counts.mean().item()})

        for T in [2, 5]:
            B = laws.our_batch(k, n, T)
            a = laws.alpha_law(k, B, n, T)
            for seed in range(10):
                on, st = emeans(Xtr, k, T, B, a, seed=seed)
                off, _ = emeans(Xtr, k, T, B, a, seed=seed, revive_every=0)
                rows.append({"dist": dist, "m": m, "T": T, "seed": seed, "revived": st["revived"],
                             "on": exact_mse(Xte, on), "off": exact_mse(Xte, off)})
        print(f"{dist} m={m} done")
        del Xtr, Xte
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df["off_vs_on_%"] = 100 * (df.off / df.on - 1)
df.to_csv(os.path.join(ROOT, "results", "revival_lowdim.csv"), index=False)
g = df.groupby(["dist", "m", "T"]).agg(revived=("revived", "mean"),
                                       diff_mean=("off_vs_on_%", "mean"),
                                       diff_sd=("off_vs_on_%", "std"),
                                       off_better=("off_vs_on_%", lambda s: int((s < 0).sum())))
g["diff_se"] = g.diff_sd / 10 ** 0.5
print("\npaired difference, revival off vs on (negative = off is better), 10 seeds")
print(g.round(3).to_string())
print("\nconverged lloyd (100 iters): clusters below the revival threshold")
print(pd.DataFrame(small).round(2).to_string(index=False))
