# quick check that things run on the gpu + rough timing per run
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


n, k = 100_000, 1024
for m in [2, 16, 128]:
    Xtr, Xva, Xte = make_data(n, m, seed=0, device=dev)
    for T in [2, 5]:
        B = laws.our_batch(k, n, T)
        a = laws.alpha_law(k, B, n, T)
        sync()
        t0 = time.perf_counter()
        C, st = emeans(Xtr, k, T, B, a, seed=0)
        sync()
        secs = time.perf_counter() - t0
        print(f"m={m:3d}  emeans T={T} B={B} alpha0={a:.3f}  test mse {exact_mse(Xte, C):.6f}"
              f"  revived {st['revived']}  {secs:.2f}s")

    sync()
    t0 = time.perf_counter()
    C, st = lloyd(Xtr, k, iters=25, seed=0)
    sync()
    secs = time.perf_counter() - t0
    print(f"m={m:3d}  lloyd@25                         test mse {exact_mse(Xte, C):.6f}"
          f"  splits {st['split']}  {secs:.2f}s")
