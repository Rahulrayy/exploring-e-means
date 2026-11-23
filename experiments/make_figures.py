# makes the figures from the csv files in results/ (skips blocks that aren't there yet)
# usage: python experiments/make_figures.py
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import per_config, per_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    path = os.path.join(RES, f"block_{name}.csv")
    if not os.path.exists(path):
        print(f"no results for block {name}, skipping")
        return None
    return pd.read_csv(path, on_bad_lines="skip")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def fig_shift(df, name, title):
    # F2: log2(alpha*/alpha_law) vs m, one line per k, one panel per T
    cfg = per_config(df)
    seeds = per_seed(df)
    Ts = sorted(cfg["T"].unique())
    fig, axes = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4), sharey=True, squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for k in sorted(cfg.k.unique()):
            c = cfg[(cfg["T"] == T) & (cfg.k == k)].sort_values("m")
            s = seeds[(seeds["T"] == T) & (seeds.k == k)]
            q = s.groupby("m")["shift"].quantile([0.25, 0.75]).unstack()
            line, = ax.plot(c.m, c.shift_ref, marker="o", label=f"k={k}")
            ax.fill_between(q.index, q[0.25], q[0.75], color=line.get_color(), alpha=0.15)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("intrinsic dim m")
        ax.set_title(f"T = {T}")
    axes[0, 0].set_ylabel("log2(alpha* / alpha_law)  [octaves]")
    axes[0, 0].legend()
    fig.suptitle(title)
    save(fig, name)


def fig_regret(df, name):
    # F3: how much mse you lose by using the law, plain and normalized
    cfg = per_config(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (k, T), c in cfg.groupby(["k", "T"]):
        c = c.sort_values("m")
        axes[0].plot(c.m, c["regret_%"], marker="o", label=f"k={k}, T={T}")
        axes[1].plot(c.m, c["nregret_%"], marker="o", label=f"k={k}, T={T}")
    axes[0].axhline(0.2, color="gray", ls=":", lw=1)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("regret (% test mse)")
    axes[1].set_ylabel("normalized regret (% of achievable gain)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("intrinsic dim m")
    axes[0].legend(fontsize=7)
    save(fig, name)


def fig_valleys(df, name, k=1024, T=2, ms=(2, 16, 128)):
    # F4: val mse vs alpha for a few m, relative to each curve's own minimum
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ms:
        sub = df[(df.m == m) & (df.k == k) & (df["T"] == T)]
        if len(sub) == 0:
            continue
        med = sub.groupby("alpha0").val.median().sort_index()
        x = np.log2(med.index.values / sub.alpha_law.iloc[0])
        ax.plot(x, 100 * (med.values / med.values.min() - 1), marker="o", ms=3, label=f"m={m}")
    ax.axvline(0, color="k", ls="--", lw=1, label="paper's law")
    ax.axhline(0.2, color="gray", ls=":", lw=1)
    ax.set_ylim(-0.2, 5)
    ax.set_xlabel("log2(alpha0 / alpha_law)")
    ax.set_ylabel("val mse above best (%)")
    ax.set_title(f"k={k}, T={T}")
    ax.legend()
    save(fig, name)


def fig_revivals(df, name):
    # F6: how often revival fires at the law's alpha
    at_law = df[df.j == 0]
    g = at_law.groupby(["k", "T", "m"]).revived.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (k, T), c in g.groupby(["k", "T"]):
        ax.plot(c.m, c.revived / k, marker="o", label=f"k={k}, T={T}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("intrinsic dim m")
    ax.set_ylabel("revivals per cluster per run")
    ax.legend(fontsize=7)
    save(fig, name)


def fig_norev(b, nr, name):
    # shift with and without revival at the same configs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, T in zip(axes, [2, 5]):
        for df, label in [(b[(b.k == 1024) & (b.m <= 16)], "revival on"), (nr, "revival off")]:
            c = per_config(df[df["T"] == T]).sort_values("m")
            ax.plot(c.m, c.shift_ref, marker="o", label=label)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("intrinsic dim m")
        ax.set_title(f"k=1024, T={T}")
    axes[0].set_ylabel("log2(alpha* / alpha_law)")
    axes[0].legend()
    save(fig, name)


if __name__ == "__main__":
    b = load("B")
    if b is not None:
        fig_shift(b, "F2_shift_vs_m.png", "best step vs the paper's law (gaussian, n=1e5)")
        fig_regret(b, "F3_regret_vs_m.png")
        fig_valleys(b, "F4_valleys.png")
        fig_revivals(b, "F6_revivals.png")
    nr = load("B_norev")
    if b is not None and nr is not None:
        fig_norev(b, nr, "F2b_shift_revival_on_off.png")
