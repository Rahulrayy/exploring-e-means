# block G: how many epochs does e-means (paper's own laws) need to beat lloyd@25, as a function of m?
# this is the same question as fig 11 in the paper, on our gaussian data
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
n = 100_000
out = os.path.join(ROOT, "results", "block_G.csv")

rows = []
for m in [2, 4, 8, 16, 32, 64, 128]:
    Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
    for k in [256, 1024, 4096]:
        for seed in range(3):
            C, _ = lloyd(Xtr, k, iters=25, seed=seed)
            rows.append({"m": m, "k": k, "seed": seed, "method": "lloyd", "T": 25,
                         "test": exact_mse(Xte, C), "train": exact_mse(Xtr, C)})
            for T in range(1, 11):
                B = laws.knee_batch(k, n, T)
                C, _ = emeans(Xtr, k, T, B, laws.alpha_law(k, B, n, T), seed=seed)
                rows.append({"m": m, "k": k, "seed": seed, "method": "emeans", "T": T,
                             "test": exact_mse(Xte, C), "train": exact_mse(Xtr, C)})
        print(f"m={m} k={k} done", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(out, index=False)

# first T where the mean over seeds beats lloyd@25
summary = []
for (m, k), sub in df.groupby(["m", "k"]):
    ref_test = sub[sub.method == "lloyd"].test.mean()
    ref_train = sub[sub.method == "lloyd"].train.mean()
    em = sub[sub.method == "emeans"].groupby("T")[["test", "train"]].mean()
    win_test = em.index[em.test < ref_test]
    win_train = em.index[em.train < ref_train]
    summary.append({"m": m, "k": k,
                    "T_test": int(win_test.min()) if len(win_test) else None,
                    "T_train": int(win_train.min()) if len(win_train) else None,
                    "best_vs_lloyd_test_%": 100 * (em.test.min() / ref_test - 1)})
print(pd.DataFrame(summary).round(3).to_string(index=False))
