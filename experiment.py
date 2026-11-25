"""Model experiment: compares ways of building the daily AI model on the real data.
Run on GitHub (experiment.yml). Prints a comparison table - changes nothing in the bot.

Variants
  A  current      : predict "coin up in 7 days?", raw inputs
  B  relative     : predict "coin beats the average coin in 7 days?" (market swings removed)
  C  relative+rank: B + every input converted to a rank among all coins that day
  D  C + ensemble : average of a tree model and a linear model

Judged with walk-forward testing: 4 separate recent periods, each predicted by a model
trained only on data from before it. Main score = "top 10% picks vs average coin": for
every day, buy the model's top 10% coins - how much better did they do than all coins?
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import core
import train

H = core.HORIZON
FOLD_DAYS, N_FOLDS, MIN_COINS = 90, 4, 40


def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=1000,
                                          l2_regularization=1.0, max_iter=200, random_state=0)


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))


def portfolio(df, p):
    """Every day: top/bottom 10% of coins by prediction vs the average coin (7-day returns)."""
    d = df[["date", "fwd_ret"]].assign(p=p)
    out = []
    for _, g in d.groupby("date"):
        if len(g) < MIN_COINS:
            continue
        k = max(int(len(g) * 0.1), 3)
        g = g.sort_values("p")
        top, bot, avg = g["fwd_ret"].iloc[-k:].mean(), g["fwd_ret"].iloc[:k].mean(), g["fwd_ret"].mean()
        out.append((top - avg, top - bot, top > avg, g["fwd_ret"].iloc[-k:].median() - g["fwd_ret"].median()))
    o = np.array(out, dtype=float)
    return {"top_vs_avg": 100 * o[:, 0].mean(), "top_vs_bottom": 100 * o[:, 1].mean(),
            "days_top_beat_avg": 100 * o[:, 2].mean(), "median_top_vs_avg": 100 * o[:, 3].mean(), "days": len(o)}


def main():
    t0 = time.time()
    data = train.load_dataset(1000, False, "binance")
    data = data.dropna(subset=["fwd_ret"])
    data = data[data["fwd_ret"].abs() > 1e-9].copy()
    cols = [c for c in core.feature_columns(data) if data[c].notna().mean() > 0.05]
    # relative label: beat the average (median) coin that day?
    data["rel"] = data["fwd_ret"] - data.groupby("date")["fwd_ret"].transform("median")
    ranked = data.groupby("date")[cols].rank(pct=True)
    ranked.columns = [f"rk_{c}" for c in cols]
    data = pd.concat([data, ranked], axis=1)
    rk_cols = list(ranked.columns)
    print(f"Data: {len(data):,} coin-days, {data['coin'].nunique()} coins, {len(cols)} inputs "
          f"({time.time() - t0:.0f}s)")

    end = data["date"].max()
    folds = [(end - pd.Timedelta(days=FOLD_DAYS * (i + 1)), end - pd.Timedelta(days=FOLD_DAYS * i))
             for i in range(N_FOLDS)][::-1]
    variants = {
        "A current (up/down, raw)": ("abs", cols, [hgb]),
        "B relative (beat avg coin)": ("rel", cols, [hgb]),
        "C relative + ranked inputs": ("rel", cols + rk_cols, [hgb]),
        "D C + ensemble": ("rel", cols + rk_cols, [hgb, logreg]),
        "E ranked inputs only + ens.": ("rel", rk_cols, [hgb, logreg]),
    }
    rows = []
    for name, (label, fc, makers) in variants.items():
        res = []
        for start, stop in folds:
            tr = data[data["date"] < start - pd.Timedelta(days=H)]
            te = data[(data["date"] >= start) & (data["date"] < stop)]
            if len(tr) > 400_000:
                tr = tr.sample(400_000, random_state=0)
            y = lambda d: ((d["fwd_ret"] if label == "abs" else d["rel"]) > 0).astype(int)
            p = np.mean([m().fit(tr[fc], y(tr)).predict_proba(te[fc])[:, 1] for m in makers], axis=0)
            r = portfolio(te, p)
            r["auc_own_label"] = roc_auc_score(y(te), p)
            r["auc_beat_avg"] = roc_auc_score((te["rel"] > 0).astype(int), p)
            res.append(r)
        avg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        worst = min(r["top_vs_avg"] for r in res)
        rows.append((name, avg, worst, [round(float(r["top_vs_avg"]), 2) for r in res]))
        print(f"  done {name} ({time.time() - t0:.0f}s)")

    print("\n=== RESULTS (4 walk-forward periods of 90 days, newest last) ===")
    print(f"{'variant':30}{'top10% vs avg':>14}{'median':>8}{'top vs bot':>11}{'days won':>9}"
          f"{'AUC beat-avg':>13}{'worst period':>13}  per period")
    for name, a, worst, per in rows:
        print(f"{name:30}{a['top_vs_avg']:>+13.2f}%{a['median_top_vs_avg']:>+7.2f}%{a['top_vs_bottom']:>+10.2f}%"
              f"{a['days_top_beat_avg']:>8.0f}%{a['auc_beat_avg']:>13.4f}{worst:>+12.2f}%  {per}")
    print("\n'top10% vs avg' = extra 7-day return of the model's top 10% coins over the average coin, "
          "averaged over every day. Positive in all 4 periods = a real, repeatable edge.")


if __name__ == "__main__":
    main()
