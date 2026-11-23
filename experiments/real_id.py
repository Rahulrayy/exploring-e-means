# intrinsic dimension of every dataset:
# same estimator everywhere, on a 1e5 subsample (deduplicated), K = 1e5 / k
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id.intrinsic_dim import all_estimates
from emeans_id.real_data import have, load_real
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
N_SUB = 100_000


def subsample(X, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = X[torch.randperm(len(X), generator=g)[:N_SUB].to(X.device)]
    return torch.unique(X, dim=0)  # duplicates break the knn estimators


rows = []
# synthetic, so the correction can be fitted on estimated m (not true m)
for dist in ["gaussian", "uniform", "student"]:
    for m in [2, 4, 8, 16, 32, 64, 128]:
        X, _, _ = make_data(N_SUB, m, dist=dist, seed=0, n_val=10, n_test=10, device=dev)
        for k in [256, 1024, 4096]:
            rows.append({"data": dist, "m": m, "k": k, "n_used": len(X), **all_estimates(X, k)})
        print(dist, m, flush=True)

for ds in ["sift", "fashion", "glove", "deep", "pixels"]:
    if not have(ds):
        print(ds, "not downloaded yet, skipping")
        continue
    Xtr, _, _ = load_real(ds, device=dev)
    X = subsample(Xtr)
    del Xtr
    for k in [256, 1024]:
        rows.append({"data": ds, "m": None, "k": k, "n_used": len(X), **all_estimates(X, k)})
    print(ds, rows[-1], flush=True)
    del X
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "id_estimates.csv"), index=False)
print(df[df.m.isna()].round(2).to_string(index=False))
