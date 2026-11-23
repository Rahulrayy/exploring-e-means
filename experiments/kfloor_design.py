# design check for the fix: m_hat with a floor on the number of neighbours, K = max(20, n_used / k).
# only uses datasets we already looked at (training + the first held-out test). results -> kfloor_design.csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id.intrinsic_dim import levina_bickel
from emeans_id.real_data import load_ann, load_pixels
from emeans_id.synthetic import make_data

dev = "cuda"
K_FLOOR = 20


def m_hat(Xtr, k, floor):
    g = torch.Generator().manual_seed(0)
    X = Xtr[torch.randperm(len(Xtr), generator=g)[:100_000].to(Xtr.device)]
    X = torch.unique(X, dim=0)
    K_old = max(3, int(round(len(X) / k)))
    K_new = max(floor, K_old)
    return levina_bickel(X, K_old), levina_bickel(X, K_new), len(X), K_old, K_new


SETS = [
    ("gaussian m=2", lambda: make_data(100_000, 2, seed=0, device=dev), True, [256, 1024, 4096]),
    ("gaussian m=3", lambda: make_data(100_000, 3, seed=0, device=dev), True, [256, 1024, 4096]),
    ("gaussian m=4", lambda: make_data(100_000, 4, seed=0, device=dev), False, [256, 1024, 4096]),
    ("pixels (training photos)", lambda: load_pixels(device=dev), True, [256, 1024, 4096]),
    ("uniform m=2", lambda: make_data(100_000, 2, dist="uniform", seed=1, device=dev), True, [512, 2048]),
    ("student-t m=2", lambda: make_data(100_000, 2, dist="student", seed=1, device=dev), True, [512, 2048]),
    ("gaussian m=3, n=3e5", lambda: make_data(300_000, 3, seed=1, device=dev), True, [512, 2048]),
    ("retina pixels", lambda: load_pixels(images=["retina"], device=dev), True, [512, 2048]),
    ("gaussian m=5", lambda: make_data(100_000, 5, seed=1, device=dev), False, [512, 2048]),
    ("sift n=5e4", lambda: load_ann("sift", n_train=50_000, device=dev), False, [512, 2048, 4096]),
    ("fashion", lambda: load_ann("fashion", n_train=1_000_000, device=dev), False, [256, 1024, 4096]),
]

rows = []
for name, loader, low, ks in SETS:
    Xtr, _, _ = loader()
    for k in ks:
        old, new, n_used, K_old, K_new = m_hat(Xtr, k, K_FLOOR)
        ok_old = (old < 3.5) == low
        ok_new = (new < 3.5) == low
        rows.append({"set": name, "low_id": low, "k": k, "n_used": n_used, "K_old": K_old, "m_old": old,
                     "K_new": K_new, "m_new": new, "ok_old": ok_old, "ok_new": ok_new})
        print(rows[-1], flush=True)
    del Xtr
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "kfloor_design.csv"), index=False)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
print(f"\ncorrect: old rule {df.ok_old.sum()}/{len(df)}, with K floor {df.ok_new.sum()}/{len(df)}")
