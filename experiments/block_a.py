# block A: do the ID estimators find the true m? (and what does noise do to them)
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from emeans_id.intrinsic_dim import all_estimates
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

n, D, k = 100_000, 128, 1024
ms = [2, 4, 8, 16, 32, 64, 128]
sigmas = [0, 0.03, 0.1, 0.3]

rows = []
for dist in ["gaussian", "student"]:
    for sigma in sigmas:
        for m in ms:
            X, _, _ = make_data(n, m, dist=dist, D=D, sigma=sigma, seed=0,
                                n_val=10, n_test=10, device=dev)
            est = all_estimates(X, k)
            rows.append({"dist": dist, "sigma": sigma, "m": m, **est})
            print(rows[-1])
            del X
            torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "block_a_id.csv"), index=False)

# F1: estimate vs true m
names = {"pr": "PCA participation ratio", "pca95": "PCA 95%", "twonn": "TwoNN",
         "lb20": "Levina-Bickel K=20", "lb_cell": f"Levina-Bickel K=n/k={round(n / k)}"}
fig, axes = plt.subplots(2, len(sigmas), figsize=(4 * len(sigmas), 7), sharex=True, sharey=True)
for i, dist in enumerate(["gaussian", "student"]):
    for j, sigma in enumerate(sigmas):
        ax = axes[i, j]
        sub = df[(df.dist == dist) & (df.sigma == sigma)]
        for col, label in names.items():
            ax.plot(sub.m, sub[col], marker="o", ms=3, label=label)
        ax.plot(ms, ms, "k--", lw=1, label="true m")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(f"{dist}, noise sigma={sigma}")
        if i == 1:
            ax.set_xlabel("true intrinsic dim m")
        if j == 0:
            ax.set_ylabel("estimated dim")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "F1_id_estimators.png"), dpi=150)
print("saved figures/F1_id_estimators.png")
