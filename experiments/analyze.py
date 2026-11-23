# turns the ladder csvs into per-config numbers (alpha*, shift, regret, valley width)
# and fits shift = a + b*log2(m) with a cluster bootstrap over (k, T)
# usage: python experiments/analyze.py B     (any block name from run_block.py)
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP = ["block", "dist", "m", "sigma", "zador", "k", "T"]
pd.set_option("display.width", 220)


def refine(x, y, i):
    # parabola through the 3 rungs around the minimum, x is log2(alpha / alpha_law)
    if i == 0 or i == len(x) - 1:
        return x[i]
    a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    if a <= 0:
        return x[i]
    return -b / (2 * a)


def per_config(df):
    rows = []
    for key, sub in df.groupby(GROUP):
        a_law = sub.alpha_law.iloc[0]
        med = sub.groupby("alpha0").agg(val=("val", "median"), test=("test", "median"),
                                        revived=("revived", "mean")).reset_index()
        med = med.sort_values("alpha0")
        x = np.log2(med.alpha0.values / a_law)
        v = med.val.values
        i = int(np.argmin(v))
        a_star = med.alpha0.values[i]

        # regret: same seed at alpha_law (rung j=0) vs at alpha*
        at_law = sub[sub.j == 0].set_index("seed")
        at_star = sub[np.isclose(sub.alpha0, a_star)].set_index("seed")
        reg = (at_law.test - at_star.test) / at_star.test
        nreg = (at_law.test - at_star.test) / (at_law.init_test - at_star.test)

        near = x[v <= v.min() * 1.002]  # rungs within +0.2% of the best
        rows.append({**dict(zip(GROUP, key)), "alpha_law": a_law, "alpha_star": a_star,
                     "shift": x[i], "shift_ref": refine(x, v, i),
                     "regret_%": 100 * reg.median(), "nregret_%": 100 * nreg.median(),
                     "width_oct": near.max() - near.min(),
                     "edge": i == 0 or (i == len(x) - 1 and a_star < 1), "capped": a_star >= 1,
                     "revived_at_law": med.revived.values[np.argmin(np.abs(x))]})
    return pd.DataFrame(rows)


def per_seed(df):
    # alpha* for every seed on its own, these go into the slope fit
    rows = []
    for key, sub in df.groupby(GROUP + ["seed"]):
        sub = sub.sort_values("alpha0")
        i = int(np.argmin(sub.val.values))
        rows.append({**dict(zip(GROUP + ["seed"], key)),
                     "shift": np.log2(sub.alpha0.values[i] / sub.alpha_law.iloc[0])})
    return pd.DataFrame(rows)


def slope(seed_df, xcol="m", n_boot=2000, seed=0):
    x = np.log2(seed_df[xcol].values.astype(float))
    y = seed_df["shift"].values
    b = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)

    # cluster bootstrap: resample whole (k, T) groups
    groups = [g for _, g in seed_df.groupby(["k", "T"])]
    boots = []
    for _ in range(n_boot):
        pick = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(np.polyfit(np.log2(pick[xcol].values.astype(float)), pick["shift"].values, 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # permutation test: shuffle m inside each (k, T) group
    perm = []
    for _ in range(n_boot):
        xs = seed_df.groupby(["k", "T"])[xcol].transform(lambda s: rng.permutation(s.values))
        perm.append(np.polyfit(np.log2(xs.values.astype(float)), y, 1)[0])
    p = float(np.mean(np.abs(perm) >= abs(b)))
    return b, lo, hi, p


def main(name):
    df = pd.read_csv(os.path.join(ROOT, "results", f"block_{name}.csv"), on_bad_lines="skip")
    cfg = per_config(df)
    cfg.to_csv(os.path.join(ROOT, "results", f"summary_{name}.csv"), index=False)
    show = ["dist", "m", "sigma", "zador", "k", "T", "alpha_law", "alpha_star", "shift",
            "shift_ref", "regret_%", "nregret_%", "width_oct", "edge", "capped"]
    print(cfg[show].round(4).to_string(index=False))

    if df.m.nunique() > 2:
        sd = per_seed(df)
        b, lo, hi, p = slope(sd)
        print(f"\nslope of shift vs log2(m): b = {b:.3f} octaves per doubling of m, "
              f"95% CI [{lo:.3f}, {hi:.3f}], permutation p = {p:.4f}")
        print(f"predicted shift from m=2 to m=128: {6 * b:.2f} octaves")


if __name__ == "__main__":
    main(sys.argv[1])
