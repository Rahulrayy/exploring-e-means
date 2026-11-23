# all the figures: method sketches + every result so far
# usage: python experiments/all_figures.py   (needs the csv files in results/)
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
from matplotlib.colors import ListedColormap

from analyze import per_config, per_seed
from plot_style import (AQUA, BLUE, BLUE_RAMP, GRAY, DARK, DARKGREY, MUTED, ORANGE, SERIES, YELLOW,
                        log2_axis, ref_line, setup)
from emeans_id import laws
from emeans_id.assign import exact_mse
from emeans_id.emeans import emeans
from emeans_id.init import kmeanspp
from emeans_id.lloyd import lloyd
from emeans_id.synthetic import make_data

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
ZADOR_M2_K1024 = 0.001968  # high-rate bound for our 2-d gaussian, k=1024 (zador high-rate formula)
K_COLORS = {256: BLUE, 1024: ORANGE, 4096: AQUA}


def res(name):
    return pd.read_csv(os.path.join(RES, name), on_bad_lines="skip")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("saved", name)


# ---------------- method pictures ----------------

def fig_schedule():
    t = np.linspace(0, 1, 400)
    a = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t))
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(t, a, color=BLUE)
    ref_line(ax, y=0.1)
    ax.text(0.99, 0.14, "floor = 0.1 · α₀", ha="right", color=DARKGREY, fontsize=8)
    ax.set_xlabel("progress through training  (t / M)")
    ax.set_ylabel("step  α(t) / α₀")
    ax.set_ylim(0, 1.08)
    ax.set_title("e-means step size: cosine from α₀ down to 0.1·α₀ (paper Eq. 5)")
    save(fig, "fig01_cosine_schedule.png")


def fig_zador():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    r = np.linspace(0, 3, 400)
    s2 = 0.5  # per-coordinate variance of our 2-d data (total variance 1)
    data = r * np.exp(-r ** 2 / (2 * s2))
    opt = r * np.exp(-r ** 2 / (4 * s2))
    data /= np.trapezoid(data, r)
    opt /= np.trapezoid(opt, r)
    ax = axes[0]
    ax.plot(r, data, color=BLUE, label="data = random init  (density p)")
    ax.plot(r, opt, color=ORANGE, label=r"best centroid density  ($p^{1/2}$)")
    ax.set_xlabel("distance from the centre  r")
    ax.set_ylabel("share of points per unit r")
    ax.set_title("2-d Gaussian: the best centroids sit further out")
    ax.legend(loc="upper right")
    ax = axes[1]
    m = 2.0 ** np.arange(1, 8)
    ax.plot(m, m / (m + 2), marker="o", color=BLUE)
    ref_line(ax, y=1)
    for mm in [2, 4, 128]:
        ax.annotate(f"{mm / (mm + 2):.2f}", (mm, mm / (mm + 2)), textcoords="offset points",
                    xytext=(0, -14), ha="center", color=DARKGREY, fontsize=8)
    log2_axis(ax)
    ax.set_ylim(0.4, 1.08)
    ax.set_xlabel("intrinsic dimension m")
    ax.set_ylabel("exponent  m / (m+2)")
    ax.set_title(r"Zador: best density $\propto p^{m/(m+2)}$, random init gives $p^1$")
    save(fig, "fig02_zador_density.png")


def fig_centroids():
    # the picture of the low-m problem: where the centroids end up on 2-d data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m, k, T = 100_000, 2, 1024, 5
    Xtr, _, Xte = make_data(n, m, seed=0, device=dev)
    B = laws.our_batch(k, n, T)
    a = laws.alpha_law(k, B, n, T)
    kpp = kmeanspp(Xtr, k, seed=0)
    runs = {
        "random init, no training": emeans(Xtr, k, 0, B, a, seed=0)[0],
        "Lloyd@25 from random init": lloyd(Xtr, k, 25, seed=0)[0],
        "Lloyd@25 from k-means++": lloyd(Xtr, k, 25, init=kpp, seed=0)[0],
        "e-means (paper, τ=0.1) from k-means++": emeans(Xtr, k, T, B, a, init=kpp, seed=0)[0],
        "e-means τ=0.01 from k-means++": emeans(Xtr, k, T, B, a, init=kpp, seed=0, tau=0.01)[0],
    }
    mses = {name: exact_mse(Xte, C) for name, C in runs.items()}
    pts = Xtr[:4000].cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), sharex=True, sharey=True)
    for ax, (name, C) in zip(axes, runs.items()):
        c = C.cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, color=GRAY, linewidths=0, label="data")
        ax.scatter(c[:, 0], c[:, 1], s=4, color=BLUE, linewidths=0, label="centroids")
        over = 100 * (mses[name] / ZADOR_M2_K1024 - 1)
        ax.set_title(f"{name}\nMSE {mses[name] * 1e3:.2f}e-3  ·  +{over:.0f}% above the bound",
                     fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.grid(False)
    axes[0].legend(loc="lower left", markerscale=3)
    fig.suptitle("Where 1024 centroids end up on 2-d Gaussian data (grey = data sample, blue = centroids)",
                 x=0.01, ha="left", color=DARK, fontsize=10)
    save(fig, "fig03_centroids_m2.png")

    # radial profile of the centroids against the two theory curves
    r = np.linspace(0, 2.6, 300)
    theory_p = r * np.exp(-r ** 2)          # data density in r (sigma^2 = 0.5)
    theory_z = r * np.exp(-r ** 2 / 2)      # p^{1/2}
    bins = np.linspace(0, 2.6, 27)
    width = bins[1] - bins[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    panels = [["random init, no training", "Lloyd@25 from random init"],
              ["Lloyd@25 from k-means++", "e-means (paper, τ=0.1) from k-means++",
               "e-means τ=0.01 from k-means++"]]
    titles = ["Random init: Lloyd stays close to the data density",
              "k-means++ start: revival (τ=0.1) pulls centroids back in"]
    for ax, names, title in zip(axes, panels, titles):
        ax.plot(r, theory_p / np.trapezoid(theory_p, r), color=DARKGREY, lw=1, ls="--", label="theory: density p")
        ax.plot(r, theory_z / np.trapezoid(theory_z, r), color=DARK, lw=1.2, label=r"theory: best ($p^{1/2}$)")
        for i, name in enumerate(names):
            rad = np.linalg.norm(runs[name].cpu().numpy(), axis=1)
            h, _ = np.histogram(rad, bins=bins)
            ax.step(bins[:-1] + width / 2, h / (h.sum() * width), where="mid", color=SERIES[i], label=name)
        ax.set_xlabel("distance of a centroid from the centre  r")
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("share of centroids per unit r")
    save(fig, "fig04_centroid_radius.png")


# ---------------- results ----------------

def fig_id():
    df = res("block_a_id.csv")
    names = {"pr": "PCA participation ratio", "pca95": "PCA 95%", "twonn": "TwoNN",
             "lb20": "Levina-Bickel K=20", "lb_cell": "Levina-Bickel K=n/k (cluster scale)"}
    sigmas = sorted(df.sigma.unique())
    fig, axes = plt.subplots(2, len(sigmas), figsize=(14, 6.4), sharex=True, sharey=True)
    for i, dist in enumerate(["gaussian", "student"]):
        for j, sigma in enumerate(sigmas):
            ax = axes[i, j]
            sub = df[(df.dist == dist) & (df.sigma == sigma)]
            ax.plot([2, 128], [2, 128], color=DARKGREY, lw=1, ls="--", label="true m")
            for c, (col, label) in enumerate(names.items()):
                ax.plot(sub.m, sub[col], marker="o", ms=4, lw=1.6, color=SERIES[c], label=label)
            log2_axis(ax)
            log2_axis(ax, "y")
            ax.set_title(f"{dist}, noise σ = {sigma}")
            if i == 1:
                ax.set_xlabel("true intrinsic dimension m")
            if j == 0:
                ax.set_ylabel("estimated dimension")
    axes[0, 0].legend(fontsize=7)
    save(fig, "fig05_id_estimators.png")


def fig_real_id():
    ids = res("id_estimates.csv")
    real = ids[ids.m.isna() & (ids.k == 1024)].sort_values("lb_cell")
    dims = {"pixels": 3, "fashion": 784, "sift": 128, "deep": 96, "glove": 100}
    labels = [f"{d}  (D = {dims[d]})" for d in real.data]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.barh(labels, real.lb_cell, color=BLUE, height=0.6)
    for y, v in enumerate(real.lb_cell):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", color=DARK, fontsize=8)
    ax.set_xlabel("estimated intrinsic dimension (Levina-Bickel, cluster scale, k=1024)")
    ax.set_title("Real datasets: few true degrees of freedom, whatever the number of columns D")
    ax.grid(axis="y", visible=False)
    save(fig, "fig06_real_id.png")


def fig_sift():
    df = res("s5_sift.csv")
    g = df.groupby(["n", "k", "method", "T"]).agg(test=("test", "mean"), secs=("secs", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for i, ((n, k), sub) in enumerate(g.groupby(["n", "k"])):
        ref = sub[sub.method == "lloyd"].test.iloc[0]
        e = sub[sub.method == "emeans"].sort_values("T")
        ax.plot(e["T"], 100 * (e.test / ref - 1), marker="o", color=SERIES[i], label=f"n={n // 1000}k, k={k}")
    ref_line(ax, y=0)
    ax.text(5, 0.1, "Lloyd@25", ha="right", color=DARKGREY, fontsize=8)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("e-means epochs (paper's own batch and step laws)")
    ax.set_ylabel("test MSE vs Lloyd@25 (%)")
    ax.set_title("SIFT: e-means passes Lloyd@25 after 2–3 epochs")
    ax.legend()

    ax = axes[1]
    rows = []
    for (n, k), sub in g[g.n == 1_000_000].groupby(["n", "k"]):
        ref = sub[sub.method == "lloyd"].iloc[0]
        e = sub[(sub.method == "emeans") & (sub.test < ref.test)].sort_values("T").iloc[0]
        rows.append((f"k={k}", ref.secs, e.secs, int(e["T"])))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r[1] for r in rows], height=0.34, color=GRAY, label="Lloyd, 25 iterations")
    ax.barh(y - 0.18, [r[2] for r in rows], height=0.34, color=BLUE, label="e-means, first epoch count that beats it")
    for i, (lab, tl, te, T) in enumerate(rows):
        ax.text(tl + 0.2, i + 0.18, f"{tl:.1f} s", va="center", fontsize=8, color=DARK)
        ax.text(te + 0.2, i - 0.18, f"{te:.1f} s  ({T} epochs, {tl / te:.0f}× faster)", va="center",
                fontsize=8, color=DARK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("GPU seconds, same assignment code for both (n = 1M)")
    ax.set_title("Same kernel, algorithm only: 8–9× faster")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig07_sift_reproduction.png")


def fig_blockg():
    df = res("block_G.csv")
    ms, ks = sorted(df.m.unique()), sorted(df.k.unique())
    M = np.full((len(ms), len(ks)), np.nan)
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            sub = df[(df.m == m) & (df.k == k)]
            ref = sub[sub.method == "lloyd"].test.mean()
            em = sub[sub.method == "emeans"].groupby("T").test.mean()
            win = em.index[em < ref]
            M[i, j] = win.min() if len(win) else np.nan
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cmap = ListedColormap(BLUE_RAMP[1:])
    im = ax.imshow(M, cmap=cmap, vmin=1, vmax=10, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(ks)):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{int(v)}", ha="center", va="center",
                    color="white" if v >= 6 else DARK, fontsize=9)
    ax.set_xticks(range(len(ks)), [f"k={k}" for k in ks])
    ax.set_yticks(range(len(ms)), [f"m={m}" for m in ms])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.set_label("epochs needed", color=DARKGREY)
    cb.outline.set_visible(False)
    ax.set_title("Epochs for e-means to beat Lloyd@25\n(paper's laws, Gaussian, n=1e5)")
    save(fig, "fig08_epochs_to_beat_lloyd.png")


def fig_shift_regret():
    b = res("block_B.csv")
    cfg = per_config(b)
    seeds = per_seed(b)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, T in zip(axes, [2, 5]):
        for k, color in K_COLORS.items():
            c = cfg[(cfg["T"] == T) & (cfg.k == k)].sort_values("m")
            q = seeds[(seeds["T"] == T) & (seeds.k == k)].groupby("m")["shift"].quantile([0.25, 0.75]).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.12, lw=0)
            ax.plot(c.m, c.shift_ref, marker="o", color=color, label=f"k={k}")
        ref_line(ax, y=0)
        ax.text(2, -0.5, "paper's law", color=DARKGREY, fontsize=8)
        log2_axis(ax)
        ax.set_xlabel("intrinsic dimension m")
        ax.set_title(f"T = {T} epochs")
    axes[0].set_ylabel("best step vs the law  log2(α*/α_law)")
    axes[0].legend()
    fig.suptitle("No dimension trend in the best step (slope −0.02, 95% CI [−0.46, 0.35]); bands = seed IQR",
                 x=0.01, ha="left", fontsize=10, color=DARK)
    save(fig, "fig09_shift_vs_m.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, T in zip(axes, [2, 5]):
        for k, color in K_COLORS.items():
            c = cfg[(cfg["T"] == T) & (cfg.k == k)].sort_values("m")
            ax.plot(c.m, c["regret_%"], marker="o", color=color, label=f"k={k}")
        ax.axhline(0.5, color=MUTED, lw=1)
        # short labels just outside the right edge so they never sit on the data
        tr = ax.get_yaxis_transform()
        ax.text(1.01, 0.5, "0.5%", transform=tr, va="center", color=DARKGREY, fontsize=8)
        ax.axhline(0.2, color=MUTED, lw=1)
        ax.text(1.01, 0.2, "0.2%", transform=tr, va="center", color=DARKGREY, fontsize=8)
        log2_axis(ax)
        ax.set_xlabel("intrinsic dimension m")
        ax.set_title(f"T = {T} epochs")
    axes[0].set_ylabel("MSE lost by using the law (%)")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    fig.suptitle("The paper's step law only costs real accuracy at m = 2   "
                 "(grey lines: 0.2% = paper's tolerance, 0.5% = our 'matters' threshold)",
                 x=0.01, ha="left", fontsize=10, color=DARK)
    save(fig, "fig10_regret_vs_m.png")


def fig_valleys():
    b = res("block_B.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, m in enumerate([2, 16, 128]):
        sub = b[(b.m == m) & (b.k == 1024) & (b["T"] == 2)]
        med = sub.groupby("alpha0").val.median().sort_index()
        x = np.log2(med.index.values / sub.alpha_law.iloc[0])
        ax.plot(x, 100 * (med.values / med.values.min() - 1), marker="o", ms=4, color=SERIES[i], label=f"m={m}")
    ref_line(ax, x=0)
    ax.text(0.1, 4.6, "paper's law", color=DARKGREY, fontsize=8)
    ax.set_ylim(-0.2, 5)
    ax.set_xlabel("step relative to the law  log2(α₀ / α_law)")
    ax.set_ylabel("validation MSE above its best (%)")
    ax.set_title("Error vs step size (k=1024, T=2): sharp at m=2, flat at high m")
    ax.legend()
    save(fig, "fig11_valleys.png")


def fig_real_valleys():
    f = res("block_F.csv")
    summ = per_config(f)
    ids = res("id_estimates.csv")
    order = ["pixels", "fashion", "sift", "deep", "glove"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    for ax, ds in zip(axes, order):
        sub = f[(f.dist == ds) & (f.k == 1024) & (f["T"] == 2)]
        med = sub.groupby("alpha0").val.median().sort_index()
        x = np.log2(med.index.values / sub.alpha_law.iloc[0])
        ax.plot(x, 100 * (med.values / med.values.min() - 1), marker="o", ms=4, color=BLUE)
        ref_line(ax, x=0)
        mh = ids[(ids.data == ds) & (ids.k == 1024)].lb_cell.iloc[0]
        reg = summ[(summ.dist == ds) & (summ.k == 1024) & (summ["T"] == 2)]["regret_%"].iloc[0]
        ax.set_title(f"{ds}  (m̂ ≈ {mh:.0f})\nlaw loses {max(reg, 0):.2f}%", fontsize=9)
        ax.set_ylim(-0.2, 4)
        ax.set_xlabel("log2(α₀ / α_law)")
    axes[0].set_ylabel("validation MSE above best (%)")
    fig.suptitle("Real data, k=1024, T=2: dashed line = the paper's step", x=0.01, ha="left",
                 fontsize=10, color=DARK)
    save(fig, "fig12_real_valleys.png")


def fig_mechanism():
    frames = []
    for name, label in [("B", "Gaussian, random init"), ("C_zador", "Gaussian, Zador init"),
                        ("C_uniform", "uniform"), ("C_student", "Student-t")]:
        df = res(f"block_{name}.csv")
        df = df[(df.k == 1024) & (df.m <= 16)]
        c = per_config(df)
        c["label"] = label
        frames.append(c)
    cfg = pd.concat(frames)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, T in zip(axes, [2, 5]):
        for i, (label, c) in enumerate(cfg[cfg["T"] == T].groupby("label", sort=False)):
            c = c.sort_values("m")
            ax.plot(c.m, c.shift_ref, marker="o", color=SERIES[i], label=label)
        ref_line(ax, y=0)
        log2_axis(ax)
        ax.set_xlabel("intrinsic dimension m")
        ax.set_title(f"T = {T} epochs, k = 1024")
    axes[0].set_ylabel("best step vs the law  log2(α*/α_law)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Mechanism blocks: a good start (Zador) wants small steps; uniform data wants big ones",
                 x=0.01, ha="left", fontsize=10, color=DARK)
    save(fig, "fig13_mechanism_shift.png")


def fig_revivals():
    b = res("block_B.csv")
    at_law = b[b.j == 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, T in zip(axes, [2, 5]):
        for k, color in K_COLORS.items():
            g = at_law[(at_law.k == k) & (at_law["T"] == T)].groupby("m").revived.mean()
            ax.plot(g.index, g.values / k, marker="o", color=color, label=f"k={k}")
        log2_axis(ax)
        ax.set_xlabel("intrinsic dimension m")
        ax.set_title(f"T = {T} epochs")
    axes[0].set_ylabel("revivals per cluster per run")
    axes[0].legend()
    fig.suptitle("Dead-cluster revival fires far more often in high dimension (at the law's step)",
                 x=0.01, ha="left", fontsize=10, color=DARK)
    save(fig, "fig14_revivals.png")


def fig_kpp():
    df = res("kpp_check.csv")
    order = ["lloyd, random", "lloyd, k-means++", "lloyd, zador", "emeans, random", "emeans, k-means++",
             "emeans, zador", "emeans a=1 no revival, random", "k-means++ alone"]
    fam = {o: (ORANGE if o.startswith("lloyd") else BLUE if o.startswith("emeans") else GRAY) for o in order}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, m in zip(axes, [2, 4, 8]):
        g = df[(df.m == m) & (df.k == 1024)].groupby("method").test.mean()
        rel = 100 * (g / g.min() - 1)
        vals = [rel[o] for o in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, color=[fam[o] for o in order], height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yy, f"{v:.1f}%", va="center", fontsize=7.5, color=DARK)
        ax.set_yticks(y, order if m == 2 else [""] * len(order))
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel("test MSE above the best method (%)")
        ax.set_title(f"m = {m}, k = 1024")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [ORANGE, BLUE, GRAY]]
    axes[1].legend(handles, ["Lloyd@25", "e-means", "init only"], loc="upper right")
    fig.suptitle("Starting points: k-means++ fixes the low-m gap for Lloyd; e-means (paper) wastes it",
                 x=0.01, ha="left", fontsize=10, color=DARK)
    save(fig, "fig15_kmeanspp_check.png")

    # headline: absolute numbers at m=2, drawn as the gap above the zador bound
    # (dots on a line from the bound, so the axis doesn't need to start at 0)
    g = df[(df.m == 2) & (df.k == 1024)].groupby("method").test.mean()
    vals = [g[o] * 1e3 for o in order]
    bound = ZADOR_M2_K1024 * 1e3
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(order))[::-1]
    for yy, v, o in zip(y, vals, order):
        ax.plot([bound, v], [yy, yy], color=fam[o], lw=2, solid_capstyle="round")
        ax.plot(v, yy, "o", ms=8, color=fam[o], mec="white", mew=1.5)
        ax.text(v + 0.02, yy, f"+{100 * (v / bound - 1):.0f}%", va="center", fontsize=8, color=DARK)
    ax.axvline(bound, color=DARK, lw=1.2, ls="--")
    ax.text(bound + 0.01, -0.75, "theoretical best (Zador)", color=DARK, fontsize=8)
    ax.set_ylim(-1.1, len(order) - 0.4)
    ax.set_yticks(y, order)
    ax.set_xlim(bound - 0.05, max(vals) * 1.08)
    ax.set_xlabel("test MSE (×10⁻³), 2-d Gaussian, k = 1024, mean of 3 seeds")
    ax.set_title("At m = 2 the usual setups sit ~40% above what is achievable")
    ax.grid(axis="y", visible=False)
    ax.legend(handles, ["Lloyd@25", "e-means", "init only"], loc="lower right")
    save(fig, "fig16_m2_vs_bound.png")


def fig_revival_kpp():
    df = res("revival_kpp.csv")
    p = df.pivot_table(index=["m", "k", "seed"], columns="method", values="test")
    d = (100 * (p["emeans law, revival off"] / p["emeans law, revival on"] - 1)).rename("d").reset_index()
    s = d.groupby(["m", "k"]).d.agg(["mean", "std"]).reset_index()
    s["se"] = s["std"] / np.sqrt(5)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ms = sorted(s.m.unique())
    for i, (k, color) in enumerate([(256, BLUE), (1024, ORANGE)]):
        sub = s[s.k == k].set_index("m").loc[ms]
        x = np.arange(len(ms)) + (i - 0.5) * 0.36
        ax.bar(x, sub["mean"], width=0.34, color=color, yerr=sub["se"], ecolor=DARKGREY, capsize=3, label=f"k={k}")
    ax.axhline(0, color=DARKGREY, lw=1)
    ax.set_xticks(range(len(ms)), [f"m={m}" for m in ms])
    ax.set_ylabel("test MSE, revival off vs on (%)")
    ax.set_title("Same k-means++ start: turning revival off helps a lot at m=2\n(5 paired seeds, ± SE)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, "fig17_revival_from_kmeanspp.png")


def fig_tau():
    syn = res("revival_tau.csv")
    real = res("revival_tau_real.csv")
    taus = [0.1, 0.03, 0.01, 0.001, 0.0]
    names = {0.1: "0.1\n(paper)", 0.03: "0.03", 0.01: "0.01", 0.001: "0.001", 0.0: "off"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    settings = [((2, "k-means++"), "m=2, k-means++ start"), ((2, "random"), "m=2, random start"),
                ((16, "random"), "m=16"), ((128, "random"), "m=128")]
    for i, ((m, start), label) in enumerate(settings):
        g = syn[(syn.m == m) & (syn.start == start)].groupby("tau").test.mean()
        ys = [100 * (g[t] / g[0.1] - 1) for t in taus]
        ax.plot(range(len(taus)), ys, marker="o", color=SERIES[i], label=label)
    ref_line(ax, y=0)
    ax.axvspan(1.6, 2.4, color=BLUE_RAMP[0], alpha=0.6, lw=0)
    ax.set_xticks(range(len(taus)), [names[t] for t in taus])
    ax.set_xlabel("revival threshold τ")
    ax.set_ylabel("test MSE vs the paper's τ = 0.1 (%)")
    ax.set_title("Synthetic Gaussian (k=1024, 3 seeds): τ = 0.01 fixes m=2, keeps m=128")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    taus_r = [0.1, 0.01, 0.0]
    for i, ds in enumerate(["pixels", "sift"]):
        g = real[real.data == ds].groupby("tau").test.mean()
        ax.plot(range(3), [100 * (g[t] / g[0.1] - 1) for t in taus_r], marker="o", color=SERIES[i], label=ds)
    ref_line(ax, y=0)
    ax.axvspan(0.6, 1.4, color=BLUE_RAMP[0], alpha=0.6, lw=0)
    ax.set_xticks(range(3), [names[t] for t in taus_r])
    ax.set_xlabel("revival threshold τ")
    ax.set_title("Real data (k=1024, 3 seeds)")
    ax.legend()
    save(fig, "fig18_revival_threshold.png")


if __name__ == "__main__":
    setup()
    os.makedirs(FIG, exist_ok=True)
    for fn in [fig_schedule, fig_zador, fig_centroids, fig_id, fig_real_id, fig_sift, fig_blockg,
               fig_shift_regret, fig_valleys, fig_real_valleys, fig_mechanism, fig_revivals,
               fig_kpp, fig_revival_kpp, fig_tau]:
        fn()
