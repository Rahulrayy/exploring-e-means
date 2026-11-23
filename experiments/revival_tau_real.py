# same tau check on real data: does tau=0.01 hurt sift (high-ish m) and help pixels (m ~ 3)?
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch

from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.real_data import load_real

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
k, T = 1024, 5

rows = []
for ds in ["pixels", "sift"]:
    Xtr, _, Xte = load_real(ds, device=dev)
    n = len(Xtr)
    B = laws.our_batch(k, n, T)
    a = laws.alpha_law(k, B, n, T)
    for seed in range(3):
        for tau in [0.1, 0.01, 0.0]:
            kw = {"revive_every": 0} if tau == 0 else {"tau": tau}
            C, st = emeans(Xtr, k, T, B, a, seed=seed, **kw)
            rows.append({"data": ds, "tau": tau, "seed": seed, "test": exact_mse(Xte, C),
                         "revived": st["revived"]})
    print(ds, "done", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "revival_tau_real.csv"), index=False)
g = df.groupby(["data", "tau"]).agg(test=("test", "mean"), revived=("revived", "mean")).reset_index()
g["vs_paper_%"] = g.groupby("data").test.transform(lambda s: 100 * (s / s.iloc[-1] - 1))
print(g.round(6).to_string(index=False))
