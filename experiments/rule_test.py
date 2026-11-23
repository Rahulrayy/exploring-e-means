# held-out test of the low-ID recipe:
#   m_hat < 3.5  ->  k-means++ start + revival tau = 0.01,  else the paper's defaults
# writes results/rule_test.csv (one row per run, resumable) and results/rule_test_mhat.csv, draws fig20
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plot_style import BLUE, DARK, MUTED, ORANGE, ref_line, setup
from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.init import kmeanspp
from emeans_id.intrinsic_dim import levina_bickel
from emeans_id.real_data import load_ann, load_pixels
from emeans_id.synthetic import make_data

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
OUT = os.path.join(ROOT, "results", "rule_test.csv")
OUT_M = os.path.join(ROOT, "results", "rule_test_mhat.csv")
THRESH = 3.5  # m_hat below this counts as low intrinsic dimension
KS, TS, SEEDS = [512, 2048], [2, 5], range(5)
ARMS = {"default": ("random", 0.1), "kpp_only": ("kpp", 0.1), "recipe": ("kpp", 0.01)}
# name, loader, low-ID by construction?  (synthetic sets use data seed 1, the training runs used seed 0)
SETS = [
    ("uniform m=2", lambda: make_data(100_000, 2, dist="uniform", seed=1, device=dev), True),
    ("student-t m=2", lambda: make_data(100_000, 2, dist="student", seed=1, device=dev), True),
    ("gaussian m=2, n=3e5", lambda: make_data(300_000, 2, seed=1, device=dev), True),
    ("gaussian m=3, n=3e5", lambda: make_data(300_000, 3, seed=1, device=dev), True),
    ("retina pixels", lambda: load_pixels(images=["retina"], device=dev), True),
    ("gaussian m=5 (control)", lambda: make_data(100_000, 5, seed=1, device=dev), False),
    ("sift n=5e4 (control)", lambda: load_ann("sift", n_train=50_000, device=dev), False),
]
FIELDS = ["set", "k", "T", "arm", "seed", "B", "alpha0", "val", "test"]


def m_hat(Xtr, k):
    # same recipe everywhere: <=1e5 rows, deduplicated, K = n_used / k
    g = torch.Generator().manual_seed(0)
    X = Xtr[torch.randperm(len(Xtr), generator=g)[:100_000].to(Xtr.device)]
    X = torch.unique(X, dim=0)
    K = max(3, int(round(len(X) / k)))
    return levina_bickel(X, K), len(X), K


def run():
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, on_bad_lines="skip")
        done = set(zip(d.set, d.k, d["T"], d.arm, d.seed))
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    mrows = []
    for name, loader, low in SETS:
        Xtr, Xva, Xte = loader()
        n = len(Xtr)
        for k in KS:
            mh, n_used, K = m_hat(Xtr, k)
            use_recipe = mh < THRESH
            mrows.append({"set": name, "k": k, "low_id": low, "m_hat": mh, "n_used": n_used, "K": K,
                          "rule_picks": "recipe" if use_recipe else "default"})
            print(f"{name} k={k}: m_hat={mh:.2f} -> {'recipe' if use_recipe else 'default'}", flush=True)
            if not (low or use_recipe):
                continue  # control correctly left alone, nothing to run
            for seed in SEEDS:
                if all((name, k, T, arm, seed) in done for T in TS for arm in ARMS):
                    continue
                kpp = kmeanspp(Xtr, k, seed=seed)
                for T in TS:
                    B = laws.our_batch(k, n, T)
                    a = laws.alpha_law(k, B, n, T)
                    for arm, (start, tau) in ARMS.items():
                        if (name, k, T, arm, seed) in done:
                            continue
                        C, _ = emeans(Xtr, k, T, B, a, init=kpp if start == "kpp" else None,
                                      seed=seed, tau=tau)
                        w.writerow({"set": name, "k": k, "T": T, "arm": arm, "seed": seed, "B": B,
                                    "alpha0": a, "val": exact_mse(Xva, C), "test": exact_mse(Xte, C)})
                        f.flush()
        del Xtr, Xva, Xte
        torch.cuda.empty_cache()
    f.close()
    pd.DataFrame(mrows).to_csv(OUT_M, index=False)


def analyze():
    pd.set_option("display.width", 200)
    mh = pd.read_csv(OUT_M)
    print("\nclassification (threshold 3.5):")
    print(mh.round(2).to_string(index=False))
    ok_low = (mh[mh.low_id].m_hat < THRESH).all()
    ok_ctrl = (mh[~mh.low_id].m_hat >= THRESH).all()

    df = pd.read_csv(OUT, on_bad_lines="skip")
    p = df.pivot_table(index=["set", "k", "T", "seed"], columns="arm", values="test").dropna()
    p["recipe"] = 100 * (p["recipe"] / p["default"] - 1)
    p["kpp_only"] = 100 * (p["kpp_only"] / p["default"] - 1)
    cfg = p.groupby(["set", "k", "T"]).agg(recipe=("recipe", "mean"), recipe_sd=("recipe", "std"),
                                          kpp_only=("kpp_only", "mean"), seeds=("recipe", "size")).reset_index()
    cfg["recipe_se"] = cfg.recipe_sd / np.sqrt(cfg.seeds)
    # where the rule picked the default, the rule's change is 0 by definition
    picks = mh.set_index(["set", "k"]).rule_picks
    cfg["rule"] = [r.recipe if picks.loc[(r.set, r.k)] == "recipe" else 0.0 for r in cfg.itertuples()]
    cfg.to_csv(os.path.join(ROOT, "results", "rule_test_summary.csv"), index=False)
    print("\n% test mse vs the paper's default (random start, tau=0.1), 5 paired seeds:")
    print(cfg[["set", "k", "T", "kpp_only", "recipe", "recipe_se", "rule"]].round(2).to_string(index=False))

    low = cfg[cfg.set.isin(mh[mh.low_id].set.unique())]
    c2 = low.rule.mean()
    worst = (low.recipe - 2 * low.recipe_se).max()
    print("\npre-registered criteria:")
    print(f"  1) classification: low-ID all < 3.5: {ok_low}, controls all >= 3.5: {ok_ctrl} "
          f"-> {'PASS' if ok_low and ok_ctrl else 'FAIL'}")
    print(f"  2) mean change over low-ID configs <= -2%: {c2:+.2f}% -> {'PASS' if c2 <= -2 else 'FAIL'}")
    print(f"  3) no low-ID config significantly worse (mean - 2SE <= +0.2%): worst {worst:+.2f}% "
          f"-> {'PASS' if worst <= 0.2 else 'FAIL'}")
    print(f"\nshare of the gain from k-means++ alone: {low.kpp_only.mean():+.2f}% of {low.recipe.mean():+.2f}%")

    # fig20
    setup()
    sets = list(dict.fromkeys(low.set))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(sets) + 1.4))
    for col, color, label, dy in [("kpp_only", ORANGE, "k-means++ only (τ = 0.1)", 0.12),
                                  ("recipe", BLUE, "recipe: k-means++ + τ = 0.01", -0.12)]:
        ys = [len(sets) - 1 - sets.index(s) + dy for s in low.set]
        ax.scatter(low[col], ys, s=30, color=color, edgecolors="white", linewidths=0.8, label=label, zorder=3)
    ref_line(ax, x=0)
    ax.axvline(0.2, color=MUTED, lw=1)
    ax.set_yticks(range(len(sets)), sets[::-1])
    ax.set_xlabel("test MSE vs the paper's default (random start, τ = 0.1), %  ·  one dot per (k, T)")
    ax.set_title("Held-out test of the low-ID recipe (k = 512 / 2048, 5 paired seeds per dot)", color=DARK)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig20_rule_test.png"))
    print("saved fig20_rule_test.png")


if __name__ == "__main__":
    if "--analyze" not in sys.argv:
        run()
    analyze()
