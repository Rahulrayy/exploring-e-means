# can a smaller revival threshold tau keep the high-m benefit of revival and stop the low-m damage?
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
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
n, k, T = 100_000, 1024, 5
B = laws.our_batch(k, n, T)
a = laws.alpha_law(k, B, n, T)
taus = [0.1, 0.03, 0.01, 0.001, 0.0]  # 0.1 is the paper, 0 = no revival

rows = []
for m in [2, 16, 128]:
    Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
    for seed in range(3):
        starts = {"random": None}
        if m == 2:
            starts["k-means++"] = kmeanspp(Xtr, k, seed=seed)
        for sname, init in starts.items():
            for tau in taus:
                kw = {"revive_every": 0} if tau == 0 else {"tau": tau}
                C, st = emeans(Xtr, k, T, B, a, init=init, seed=seed, **kw)
                rows.append({"m": m, "start": sname, "tau": tau, "seed": seed,
                             "test": exact_mse(Xte, C), "revived": st["revived"]})
    print(f"m={m} done", flush=True)
    del Xtr, Xte
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "revival_tau.csv"), index=False)
g = df.groupby(["m", "start", "tau"]).agg(test=("test", "mean"), revived=("revived", "mean")).reset_index()
g["vs_paper_%"] = g.groupby(["m", "start"]).test.transform(lambda s: 100 * (s / s.iloc[-1] - 1))
pd.set_option("display.width", 200)
print(g.round(5).to_string(index=False))
