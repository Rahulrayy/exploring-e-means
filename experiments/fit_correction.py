# fits the dimension correction on synthetic data (block B) and tests it on real data (block F)
#   M0: shift = a
#   M1: shift = a + b*log2(m_hat)
#   M2: shift = a + b*(2/m_hat)*log2(k)      (zador-ish: log of the high-rate distortion)
# shift = log2(alpha*/alpha_law). m_hat = levina-bickel at cluster scale (the primary estimator).
# regret of a predicted shift is read off the measured median test curve (interpolated in log alpha)
import os

import numpy as np
import pandas as pd

from analyze import per_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EST = "lb_cell"
pd.set_option("display.width", 200)


def curve(sub):
    med = sub.groupby("alpha0")[["val", "test"]].median().sort_index()
    return np.log2(med.index.values / sub.alpha_law.iloc[0]), med


def regret_at(sub, shift):
    # % test mse above the ladder's best (picked on val) if we ran at this shift
    x, med = curve(sub)
    best = med.test.values[np.argmin(med.val.values)]
    return 100 * (np.interp(shift, x, med.test.values) / best - 1)


def features(model, m_hat, k):
    one = np.ones_like(m_hat, dtype=float)
    if model == "M0":
        return one[:, None]
    if model == "M1":
        return np.column_stack([one, np.log2(m_hat)])
    if model == "M2":
        return np.column_stack([one, 2 / m_hat * np.log2(k)])


def fit(model, d):
    coef, *_ = np.linalg.lstsq(features(model, d.m_hat.values, d.k.values), d["shift"].values, rcond=None)
    return coef


def predict(model, coef, m_hat, k):
    return features(model, np.atleast_1d(np.asarray(m_hat, dtype=float)),
                    np.atleast_1d(np.asarray(k, dtype=float))) @ coef


def main():
    ids = pd.read_csv(os.path.join(RES, "id_estimates.csv"))
    b = pd.read_csv(os.path.join(RES, "block_B.csv"), on_bad_lines="skip")
    syn_id = ids[ids.data == "gaussian"][["m", "k", EST]].rename(columns={EST: "m_hat"})

    seeds = per_seed(b).merge(syn_id, on=["m", "k"])
    print(f"block B: {seeds[['m', 'k', 'T']].drop_duplicates().shape[0]} configs, m = {sorted(seeds.m.unique())}")

    # fits on everything
    fits = {}
    for model in ["M0", "M1", "M2"]:
        fits[model] = fit(model, seeds)
        print(model, "coef", np.round(fits[model], 3))

    # leave one m out
    rows = []
    for m_out in sorted(seeds.m.unique()):
        train = seeds[seeds.m != m_out]
        for model in ["M0", "M1", "M2"]:
            coef = fit(model, train)
            for (k, T), sub in b[b.m == m_out].groupby(["k", "T"]):
                mh = syn_id[(syn_id.m == m_out) & (syn_id.k == k)].m_hat.iloc[0]
                pred = predict(model, coef, mh, k)[0]
                rows.append({"m_out": m_out, "k": k, "T": T, "model": model, "pred_shift": pred,
                             "regret_%": regret_at(sub, pred)})
        for (k, T), sub in b[b.m == m_out].groupby(["k", "T"]):
            rows.append({"m_out": m_out, "k": k, "T": T, "model": "paper", "pred_shift": 0.0,
                         "regret_%": regret_at(sub, 0.0)})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(RES, "correction_cv.csv"), index=False)
    print("\nleave-one-m-out, median regret (%) on the held-out m:")
    print(cv.pivot_table(index="m_out", columns="model", values="regret_%", aggfunc="median").round(3))
    print(cv.groupby("model")["regret_%"].median().round(3).to_string())

    # real data
    path_f = os.path.join(RES, "block_F.csv")
    if not os.path.exists(path_f):
        print("\nno block F yet")
        return
    f = pd.read_csv(path_f, on_bad_lines="skip")
    real_id = ids[ids.m.isna()][["data", "k", EST]].rename(columns={"data": "dist", EST: "m_hat"})
    f_seed = per_seed(f)
    if "deep" not in set(f.dist):
        print("\nno deep in block F yet, can't anchor")
        return
    # calibration: the shift we measure on deep, per (k, T)
    s_deep = f_seed[f_seed.dist == "deep"].groupby(["k", "T"])["shift"].median()
    mh_deep = real_id[real_id.dist == "deep"].set_index("k").m_hat

    rows = []
    for (ds, k, T), sub in f.groupby(["dist", "k", "T"]):
        mh = real_id[(real_id.dist == ds) & (real_id.k == k)].m_hat.iloc[0]
        x, med = curve(sub)
        s_cal = s_deep.loc[(k, T)]
        row = {"dataset": ds, "k": k, "T": T, "m_hat": mh,
               "measured_shift": x[np.argmin(med.val.values)],
               "paper": regret_at(sub, 0.0), "calibrated": regret_at(sub, s_cal)}
        for model in ["M1", "M2"]:
            # only the slope comes from synthetic data, the reference point is deep
            delta = predict(model, fits[model], mh, k)[0] - predict(model, fits[model], mh_deep.loc[k], k)[0]
            row[f"cal+{model}"] = regret_at(sub, s_cal + delta)
        rows.append(row)
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RES, "correction_real.csv"), index=False)
    print("\nreal data, regret (% test mse above the tuned optimum). deep is the anchor, so it's in-sample")
    print(ev.round(3).to_string(index=False))
    held = ev[ev.dataset != "deep"]
    print("\nmedian over the held-out datasets:")
    print(held[["paper", "calibrated", "cal+M1", "cal+M2"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
