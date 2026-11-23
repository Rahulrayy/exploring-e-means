# reads results/tau_confirm.csv, checks the pre-registered criteria for tau = 0.01
# and draws fig19
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import BLUE, DARK, DARKGREY, MUTED, ORANGE, ref_line, setup

KEY = ["data", "m", "k", "T", "start"]
pd.set_option("display.width", 200)


def label(row):
    return f"gaussian m={row.m}" if row.data == "gaussian" else row.data


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "tau_confirm.csv"), on_bad_lines="skip")
    p = df.pivot_table(index=KEY + ["seed"], columns="tau", values="test").dropna()
    p["d001"] = 100 * (p[0.01] / p[0.1] - 1)
    p["doff"] = 100 * (p[0.0] / p[0.1] - 1)
    cfg = p.groupby(KEY).agg(d001=("d001", "mean"), d001_sd=("d001", "std"),
                             doff=("doff", "mean"), seeds=("d001", "size")).reset_index()
    cfg["d001_se"] = cfg.d001_sd / np.sqrt(cfg.seeds)
    cfg["label"] = cfg.apply(label, axis=1)
    cfg.to_csv(os.path.join(ROOT, "results", "tau_confirm_summary.csv"), index=False)

    print(f"{len(cfg)} configurations, {len(p)} paired seeds")
    print("\nmean % change vs tau=0.1 (negative = better), averaged over k, T and start:")
    print(cfg.groupby("label")[["d001", "doff"]].mean().round(2).to_string())

    # pre-registered criteria
    low = cfg[((cfg.data == "gaussian") & (cfg.m <= 3)) | (cfg.data == "pixels")]
    c1 = cfg.d001.median()
    c2 = cfg.d001.max()
    c3 = low.d001.mean()
    print("\ncriteria for tau = 0.01:")
    print(f"  1) median change over all configs <= 0      : {c1:+.3f}%  -> {'PASS' if c1 <= 0 else 'FAIL'}")
    print(f"  2) no config worse than +0.2%               : worst {c2:+.3f}%  -> {'PASS' if c2 <= 0.2 else 'FAIL'}")
    print(f"  3) low-ID (gaussian m<=3, pixels) <= -1%     : {c3:+.3f}%  -> {'PASS' if c3 <= -1 else 'FAIL'}")
    print("\nworst 8 configs for tau = 0.01:")
    print(cfg.sort_values("d001", ascending=False).head(8)[KEY + ["d001", "d001_se", "doff"]].round(3).to_string(index=False))

    # fig19: one dot per (k, T) config, rows sorted roughly by intrinsic dimension
    order = ["gaussian m=2", "gaussian m=3", "pixels", "gaussian m=4", "gaussian m=8", "fashion",
             "gaussian m=16", "sift", "deep", "glove", "gaussian m=128"]
    order = [o for o in order if o in set(cfg.label)]
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(order) + 1.6), sharey=True)
    for ax, start, title in zip(axes, ["random", "kpp"], ["random start", "k-means++ start"]):
        sub = cfg[cfg.start == start]
        for col, color, name, dy in [("d001", BLUE, "τ = 0.01", -0.13), ("doff", ORANGE, "revival off", 0.13)]:
            ys = [len(order) - 1 - order.index(l) + dy for l in sub.label]
            ax.scatter(sub[col], ys, s=26, color=color, edgecolors="white", linewidths=0.8, label=name, zorder=3)
        ref_line(ax, x=0)
        ax.axvline(0.2, color=MUTED, lw=1)
        ax.set_yticks(range(len(order)), order[::-1])
        ax.set_xlabel("test MSE vs the paper's τ = 0.1 (%)  ·  one dot per (k, T)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left")
    fig.suptitle("Revival threshold check, 5 paired seeds per dot (rows sorted by intrinsic dimension, "
                 "grey line = +0.2% tolerance)", x=0.01, ha="left", fontsize=10, color=DARK)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig19_tau_confirm.png"))
    print("saved fig19_tau_confirm.png")


if __name__ == "__main__":
    main()
