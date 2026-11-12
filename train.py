"""Train the analyst.

  python train.py                      # Binance: every coin incl. delisted ones, ~2.7 years each
  python train.py --source coingecko   # CoinGecko top 1000, 1 year each (slower, rate-limited)
  python train.py --demo               # made-up data, runs in a minute, results mean nothing

Data the model learns from (Binance source): daily close/high/low, USD volume, number of
trades, aggressive-buyer volume, coin age, strength vs Bitcoin, the Crypto Fear & Greed
Index, plus the 8 strategy signals. Delisted coins are included on purpose so the dead
ones count too (otherwise the past looks better than it was).

What "training" means here, honestly:
  * Every coin's history is cut into days. For each day we record what the 8
    strategies said and ~15 indicators, then what the price actually did over the
    next 7 days.
  * Strategies are back-tested: how often was each one right, and how much did it make?
  * A machine-learning model (gradient boosted trees vs. logistic regression - whichever
    scores better) learns how to combine all of it into "chance the price is higher in 7 days".
  * The most recent 25% of days are HIDDEN during training and used only for the final exam,
    so the accuracy you see is what it scored on data it had never seen.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import core


MIN_AUC = 0.53  # below this the model is basically guessing and gets switched off


def load_dataset(n_coins: int, demo: bool, source: str = "binance") -> pd.DataFrame:
    if demo:
        coins = [{"id": "bitcoin"}] + [{"id": f"demo-coin-{i}"} for i in range(1, n_coins)]
        get = lambda i, cid: core.fake_history(i, days=1000)
    elif source == "binance":
        syms = core.binance_coins(include_delisted=True)
        print(f"Binance: {len(syms)} coins (including delisted ones), up to 1000 days each")
        syms = ["BTC"] + [x for x in syms if x != "BTC"][:n_coins - 1]
        coins = [{"id": x} for x in syms]

        def get(i, sym):
            # Cached ~20h, so a daily retrain always learns from fresh prices.
            return core.binance_history(sym, max_age_h=20)
    else:
        print(f"Getting the top {n_coins} coins from CoinGecko...")
        coins = core.top_coins(n_coins)
        if not any(c["id"] == "bitcoin" for c in coins):
            coins.insert(0, {"id": "bitcoin"})

        def get(i, cid):
            cached = (core.DATA_DIR / f"{cid}.json").exists()
            df = core.history(cid, max_age_h=24 * 6)
            if not cached:
                time.sleep(2.2 if core.CG_KEY else 6)  # respect the free rate limit
            return df

    btc = get(0, "bitcoin")
    frames = []
    for i, c in enumerate(coins):
        try:
            df = btc if c["id"] in ("bitcoin", "BTC") else get(i, c["id"])
            listed = df.attrs.get("listed")
            df = df[(df["close"] > 0) & (df["volume"] > 0)]
            df.attrs["listed"] = listed
            if len(df) < 120:  # too new to learn from
                continue
            feat = core.add_labels(core.build_features(df, btc))
            feat["coin"] = c["id"]
            frames.append(feat)
        except Exception as e:
            print(f"  skip {c['id']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(coins)} coins loaded")
    data = pd.concat(frames).rename_axis("date").reset_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    # Clip insane returns (rug pulls, bad data) so a single coin can't dominate.
    data["fwd_ret"] = data["fwd_ret"].clip(-0.9, 3)
    return data


def backtest(df: pd.DataFrame) -> dict:
    """For each strategy: when it gave a signal, was the next 7 days in its favour?"""
    out = {}
    for name in core.STRATEGIES:
        sig = df[f"sig_{name}"]
        active = df[sig != 0]
        if len(active) == 0:
            continue
        pnl = np.sign(active[f"sig_{name}"]) * active["fwd_ret"]
        out[name] = {
            "signals": int(len(active)),
            "win_rate": round(float((pnl > 0).mean()) * 100, 1),
            "avg_return_per_signal": round(float(pnl.mean()) * 100, 2),
            "median_return_per_signal": round(float(pnl.median()) * 100, 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", type=int, default=1000)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--source", choices=["binance", "coingecko"], default="binance")
    args = ap.parse_args()
    core.OFFLINE = args.demo

    data = load_dataset(args.coins, args.demo, args.source)
    labeled = data.dropna(subset=["fwd_ret"])
    labeled = labeled[labeled["fwd_ret"].abs() > 1e-9]
    cols = core.feature_columns(labeled)
    # Skip inputs that are (almost) empty - e.g. a data source that was down today.
    empty = [c for c in cols if labeled[c].notna().mean() < 0.05]
    if empty:
        print(f"Not using (no data this time): {', '.join(empty)}")
    cols = [c for c in cols if c not in empty]
    n_coins = labeled["coin"].nunique()
    print(f"\nDataset: {len(labeled):,} coin-days from {n_coins} coins, {len(cols)} inputs each")

    # --- split by DATE: train on the past, test on the most recent stretch ---
    dates = np.sort(labeled["date"].unique())
    test_start = dates[int(len(dates) * 0.75)]
    gap = pd.Timedelta(days=core.HORIZON)  # stop labels from peeking into the test period
    train = labeled[labeled["date"] < test_start - gap]
    test = labeled[labeled["date"] >= test_start]
    tr_dates = np.sort(train["date"].unique())
    val_start = tr_dates[int(len(tr_dates) * 0.8)]
    fit = train[train["date"] < val_start - gap]
    val = train[train["date"] >= val_start]
    y = lambda d: (d["fwd_ret"] > 0).astype(int)

    # --- try a few models, keep the best on the validation set ---
    candidates = {
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=200,
            l2_regularization=1.0, random_state=0),
        "gradient_boosting_small": lambda: HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=500, random_state=0),
        "logistic_regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
    }
    print("\nModel try-outs (validation AUC, 0.50 = coin flip):")
    best_name, best_auc = None, -1
    for name, make in candidates.items():
        m = make().fit(fit[cols], y(fit))
        auc = roc_auc_score(y(val), m.predict_proba(val[cols])[:, 1])
        print(f"  {name:25} {auc:.4f}")
        if auc > best_auc:
            best_name, best_auc = name, auc

    # --- final exam on the hidden test period ---
    model = candidates[best_name]().fit(train[cols], y(train))
    p = model.predict_proba(test[cols])[:, 1]
    yt = y(test)
    base_rate = yt.mean()
    exam = {
        "model": best_name,
        "test_period": f"{pd.Timestamp(test_start).date()} to {pd.Timestamp(dates[-1]).date()}",
        "test_rows": int(len(test)),
        "accuracy": round(accuracy_score(yt, p > 0.5) * 100, 1),
        "accuracy_if_you_always_guessed_the_majority": round(max(base_rate, 1 - base_rate) * 100, 1),
        "auc": round(roc_auc_score(yt, p), 4),
    }
    # How right is it when it's confident?
    buckets = []
    for lo, hi in [(0, .35), (.35, .45), (.45, .55), (.55, .65), (.65, 1.01)]:
        mask = (p >= lo) & (p < hi)
        if mask.sum() >= 30:
            buckets.append({"model_said": f"{lo:.0%}-{min(hi, 1):.0%} chance up",
                            "times": int(mask.sum()),
                            "actually_went_up": round(float(yt[mask].mean()) * 100, 1),
                            "avg_7d_return": round(float(test["fwd_ret"][mask].mean()) * 100, 2)})
    exam["confidence_check"] = buckets

    # Strategy trust comes from the MOST RECENT months, because what works in crypto
    # changes (trending vs. choppy markets). A strategy only gets trust if its typical
    # (median) trade made money - not just its average, which a few pumps can inflate.
    bt_train, bt_test = backtest(train), backtest(test)
    weights = {k: (max(v["avg_return_per_signal"], 0) * (v["win_rate"] / 100)
                   if v["median_return_per_signal"] > 0 and v["signals"] >= 200 else 0)
               for k, v in bt_test.items()}
    total = sum(weights.values()) or 1
    weights = {k: round(v / total, 3) for k, v in weights.items()}

    # Retrain on everything so the live model knows the most recent market.
    final_model = candidates[best_name]().fit(labeled[cols], y(labeled))
    importance = {}
    if hasattr(final_model, "feature_importances_"):
        importance = dict(zip(cols, final_model.feature_importances_))

    # If the model couldn't beat a coin flip on the exam, the analyst won't listen to it.
    model_trusted = exam["auc"] >= MIN_AUC
    exam["model_trusted"] = model_trusted
    joblib.dump({"model": final_model, "columns": cols, "strategy_weights": weights,
                 "model_trusted": model_trusted, "auc": exam["auc"],
                 "demo": args.demo}, core.ROOT / "model.joblib")
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "demo_data": args.demo,
        "coins": n_coins,
        "coin_days": int(len(labeled)),
        "data_source": "demo" if args.demo else args.source,
        "inputs": cols,
        "exam": exam,
        "strategy_backtest_test_period": bt_test,
        "strategy_weights": weights,
        "buy_and_hold_avg_7d_return_test": round(float(test["fwd_ret"].mean()) * 100, 2),
    }
    (core.ROOT / "training_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== FINAL EXAM ({exam['test_period']}, never seen during training) ===")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {exam['accuracy']}%  (always guessing the majority would get "
          f"{exam['accuracy_if_you_always_guessed_the_majority']}%)   AUC {exam['auc']}")
    print("When the model said...")
    for b in buckets:
        print(f"  {b['model_said']:18} -> went up {b['actually_went_up']}% of the time "
              f"({b['times']:,} cases, avg 7d {b['avg_7d_return']:+}%)")
    if not model_trusted:
        print(f"!! AUC {exam['auc']} is below {MIN_AUC}: the model has no real edge, so it is "
              "SWITCHED OFF. Verdicts will use only the strategies that worked recently.")
    print("\nStrategy scoreboard (test period):")
    print(f"  {'strategy':22}{'signals':>9}{'win rate':>10}{'avg 7d':>9}")
    for k, v in sorted(bt_test.items(), key=lambda kv: -kv[1]["avg_return_per_signal"]):
        print(f"  {k:22}{v['signals']:>9,}{v['win_rate']:>9}%{v['avg_return_per_signal']:>+8}%")
    print(f"  (buy-and-hold everything averaged {report['buy_and_hold_avg_7d_return_test']:+}% per 7 days)")
    print("Strategy trust now:", {k: v for k, v in weights.items() if v})
    print("\nSaved model.joblib and training_report.json")
    if args.demo:
        print("\n!! DEMO MODE: trained on fake prices. These numbers say nothing about real crypto.")


if __name__ == "__main__":
    main()
