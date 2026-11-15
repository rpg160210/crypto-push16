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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import core


MIN_AUC = 0.53  # below this the model is basically guessing and gets switched off
MIN_IMPROVEMENT = 0.003  # a challenger must beat the champion by this much to replace it
KEEP_BEST = 3            # best settings from earlier runs that get re-tested every run
LEADERBOARD_SIZE = 15
SEARCH_ROWS = 250_000    # rows used while searching settings (keeps runs fast)

# Groups of inputs a model may use. Trying different groups helps find out which
# data actually helps and which is just noise.
MARKET = ("btc_", "fear_greed", "vs_btc")
EXTRA = ("atr_pct", "close_in_range", "taker_buy", "trades_ratio", "coin_age_days",
         "ret_90d", "drawdown_365d")
FEATURE_SETS = {
    "all": lambda c: True,
    "no_strategy_signals": lambda c: not c.startswith("sig_"),
    "no_market_mood": lambda c: not c.startswith(MARKET),
    "price_volume_only": lambda c: not c.startswith(MARKET + EXTRA),
}
STARTERS = [
    {"kind": "logreg", "C": 0.1, "features": "all"},
    {"kind": "hgb", "learning_rate": 0.05, "max_leaf_nodes": 8, "min_samples_leaf": 500,
     "l2": 1.0, "max_iter": 150, "max_features": 1.0, "features": "all"},
    {"kind": "hgb", "learning_rate": 0.04, "max_leaf_nodes": 24, "min_samples_leaf": 200,
     "l2": 1.0, "max_iter": 300, "max_features": 1.0, "features": "all"},
]


def random_config(rng) -> dict:
    """A brand-new combination of model settings to try."""
    features = str(rng.choice(list(FEATURE_SETS)))
    if rng.random() < 0.3:
        return {"kind": "logreg", "C": round(float(10 ** rng.uniform(-2.5, 0.5)), 4),
                "features": features}
    return {"kind": "hgb",
            "learning_rate": round(float(10 ** rng.uniform(-1.8, -0.8)), 4),
            "max_leaf_nodes": int(rng.choice([7, 15, 24, 31, 48])),
            "min_samples_leaf": int(rng.choice([100, 200, 500, 1000, 2000])),
            "l2": round(float(10 ** rng.uniform(-1, 1)), 3),
            "max_iter": int(rng.choice([100, 200, 300, 500])),
            "max_features": float(rng.choice([0.5, 0.7, 1.0])),
            "features": features}


def make_model(cfg):
    if cfg["kind"] == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=cfg["C"], max_iter=1000))
    return HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=cfg["l2"],
        max_iter=cfg["max_iter"], max_features=cfg["max_features"],
        early_stopping=False, random_state=0)


def features_for(cfg, cols):
    keep = FEATURE_SETS.get(cfg.get("features", "all"), FEATURE_SETS["all"])
    return [c for c in cols if keep(c)]


CALL_WEIGHT = 5  # its own past calls count 5x in training


def review_past_calls(demo):
    """Check every call the bot made at least 7 days ago against what the price did next.
    Buy calls are right if the price went up; sell/avoid calls are right if it went down."""
    if demo:
        return None, pd.DataFrame()
    calls = []
    for kind, f in core.CALL_LOGS.items():
        if f.exists():
            try:
                calls += [{**c, "kind": kind} for c in json.loads(f.read_text())]
            except Exception as e:
                print(f"  couldn't read {f.name}: {e}")
    if not calls:
        return None, pd.DataFrame()
    now = pd.Timestamp.now()
    horizon = pd.Timedelta(days=core.REVIEW_DAYS)
    due = [c for c in calls if pd.to_datetime(c["time"], unit="ms") + horizon <= now]
    prices = {}
    for key in {(c["source"], c["coin"]) for c in due}:
        src, coin = key
        try:
            if src == "binance":
                start = min(c["time"] for c in due if c["coin"] == coin) - 86_400_000
                rows = core._binance_get("klines", {"symbol": f"{coin}USDT", "interval": "1d",
                                                    "startTime": int(start), "limit": 1000})
                prices[key] = pd.Series([float(r[4]) for r in rows],
                                        index=pd.to_datetime([r[0] for r in rows], unit="ms"))
            else:
                prices[key] = core.history(coin, max_age_h=20)["close"]
        except Exception as e:
            print(f"  couldn't get prices to review {coin}: {e}")

    btc = None  # benchmark: what just holding Bitcoin did over the same 7 days
    if due:
        try:
            start = min(c["time"] for c in due) - 86_400_000
            k = core._binance_get("klines", {"symbol": "BTCUSDT", "interval": "1d",
                                             "startTime": int(start), "limit": 1000})
            btc = pd.Series([float(r[4]) for r in k], index=pd.to_datetime([r[0] for r in k], unit="ms"))
        except Exception as e:
            print(f"  couldn't get Bitcoin prices for the comparison: {e}")

    results, rows = [], []
    for c in due:
        series = prices.get((c["source"], c["coin"]))
        t0 = pd.to_datetime(c["time"], unit="ms")
        target = (t0 + horizon).normalize()
        if series is None or series.index.max() < target or not c.get("price"):
            continue
        ret = float(series.asof(target)) / c["price"] - 1
        buy = c["call"] in core.BUY_CALLS
        right = ret > 0 if buy else ret < 0
        btc_ret = (float(btc.asof(target)) / float(btc.asof(t0.normalize())) - 1
                   if btc is not None and btc.index.max() >= target else None)
        results.append({"kind": c["kind"], "call": c["call"], "ret": ret, "right": right,
                        "buy": buy, "btc_ret": btc_ret})
        if c.get("features"):
            rows.append({**c["features"], "date": t0.normalize(), "coin": c["coin"], "fwd_ret": ret})

    by = {}
    for r in results:
        by.setdefault(f"{r['kind']}:{r['call']}", []).append(r)
    review = {
        "reviewed": len(results),
        "waiting_for_7_days": len(calls) - len(due),
        "overall_right_pct": round(100 * np.mean([r["right"] for r in results]), 1) if results else None,
        "by_call": {k: summarize_calls(v) for k, v in sorted(by.items())},
    }
    fb = pd.DataFrame(rows)
    if len(fb):
        fb["fwd_ret"] = fb["fwd_ret"].clip(-0.9, 3)
    return review, fb


def summarize_calls(v):
    out = {"calls": len(v), "right": int(sum(r["right"] for r in v)),
           "right_pct": round(100 * np.mean([r["right"] for r in v]), 1),
           "avg_7d_return_pct": round(100 * float(np.mean([r["ret"] for r in v])), 2)}
    b = [r for r in v if r["btc_ret"] is not None]
    if b:
        out["btc_same_weeks_pct"] = round(100 * float(np.mean([r["btc_ret"] for r in b])), 2)
        if v[0]["buy"]:  # did buying the coin beat just buying Bitcoin?
            out["beat_btc"] = int(sum(r["ret"] > r["btc_ret"] for r in b))
            out["beat_btc_of"] = len(b)
    return out


def print_review(review):
    if not review:
        print("\nPast-call review: no calls logged yet (they appear after analyze/hype run).")
        return
    print(f"\n=== REVIEW OF MY PAST CALLS (judged 7 days later) ===")
    print(f"Reviewed: {review['reviewed']}   still waiting for 7 days: {review['waiting_for_7_days']}"
          + (f"   overall right: {review['overall_right_pct']}%" if review["overall_right_pct"] is not None else ""))
    for k, v in review["by_call"].items():
        kind, call = k.split(":", 1)
        extra = ""
        if "btc_same_weeks_pct" in v:
            extra = f"   BTC same weeks {v['btc_same_weeks_pct']:+}%"
            if "beat_btc" in v:
                extra += f"   beat BTC {v['beat_btc']}/{v['beat_btc_of']}"
        print(f"  {kind:8}{call:16}{v['right']:>4}/{v['calls']:<4} right ({v['right_pct']}%)   "
              f"avg 7d after the call {v['avg_7d_return_pct']:+}%{extra}")
    print("  (buy calls are right if the price rose; sell/avoid calls if it fell. If buy calls")
    print("   don't beat BTC, simply holding Bitcoin would have been better.)")


def notify_review(review):
    lines = [f"{k.split(':', 1)[1]} ({k.split(':', 1)[0]}): {v['right']}/{v['calls']} right, "
             f"avg {v['avg_7d_return_pct']:+}% in 7d"
             + (f", beat BTC {v['beat_btc']}/{v['beat_btc_of']}" if "beat_btc" in v else "")
             for k, v in sorted(review["by_call"].items(), key=lambda kv: -kv[1]["calls"])[:8]]
    if lines:
        core.notify(f"My track record: {review['overall_right_pct']}% of {review['reviewed']} calls right",
                    "\n".join(lines) + "\n\nThe model now trains on these results too.", tags="bar_chart")


def describe(cfg) -> str:
    if cfg["kind"] == "logreg":
        return f"logistic C={cfg['C']} [{cfg['features']}]"
    return (f"trees lr={cfg['learning_rate']} leaves={cfg['max_leaf_nodes']} "
            f"min_leaf={cfg['min_samples_leaf']} l2={cfg['l2']} iters={cfg['max_iter']} "
            f"feat={cfg['max_features']} [{cfg['features']}]")


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

    btc = get(0, "BTC" if (source == "binance" and not demo) else "bitcoin")
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
    ap.add_argument("--explore", type=int, default=4, help="new random model settings to try per run")
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

    # --- Review the bot's own past calls and learn from them ---
    review, feedback = review_past_calls(args.demo)
    if len(feedback):
        fb = feedback.reindex(columns=labeled.columns)
        fb = pd.concat([fb] * CALL_WEIGHT, ignore_index=True)   # its own calls count 5x
        # Only calls from before each period may be used for that period (keeps exams honest).
        fit = pd.concat([fit, fb[fb["date"] < val_start - gap]], ignore_index=True)
        train = pd.concat([train, fb[fb["date"] < test_start - gap]], ignore_index=True)
        labeled_final = pd.concat([labeled, fb], ignore_index=True)
        print(f"Learning from {len(feedback)} of its own past calls (each counts {CALL_WEIGHT}x)")
    else:
        labeled_final = labeled

    # --- Champion vs. challengers: every run searches for a better model, but only
    #     replaces the current one if it clearly wins on the same unseen exam. ---
    prev = {}
    rp = core.ROOT / "training_report.json"
    if rp.exists():
        try:
            prev = json.loads(rp.read_text())
        except Exception:
            prev = {}
    leaderboard = prev.get("leaderboard", [])
    old_bundle = None
    if (core.ROOT / "model.joblib").exists():
        try:
            old_bundle = joblib.load(core.ROOT / "model.joblib")
        except Exception:
            old_bundle = None

    rng = np.random.default_rng()  # different every run -> keeps exploring new settings
    search_fit = fit.sample(min(len(fit), SEARCH_ROWS), random_state=0)
    candidates = [c["config"] for c in leaderboard[:KEEP_BEST]]  # best settings found so far
    if not candidates:
        candidates = list(STARTERS)
    candidates += [random_config(rng) for _ in range(args.explore)]  # brand-new ideas

    print(f"\nSearching {len(candidates)} model settings "
          f"({min(len(leaderboard), KEEP_BEST)} best-so-far + {args.explore} new). "
          "Validation AUC, 0.50 = coin flip:")
    scored = []
    for cfg in candidates:
        fc = features_for(cfg, cols)
        try:
            m = make_model(cfg).fit(search_fit[fc], y(search_fit))
            auc = roc_auc_score(y(val), m.predict_proba(val[fc])[:, 1])
        except Exception as e:
            print(f"  failed {describe(cfg)}: {e}")
            continue
        print(f"  {auc:.4f}  {describe(cfg)}")
        scored.append((auc, cfg))
    scored.sort(key=lambda t: -t[0])
    challenger_cfg = scored[0][1]

    # Update the memory of good settings (re-scored on today's data each run).
    seen, new_lb = set(), []
    for auc, cfg in scored:
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            new_lb.append({"config": cfg, "val_auc": round(float(auc), 4)})
    leaderboard = new_lb[:LEADERBOARD_SIZE]

    # Exam: challenger vs. champion, both trained on the same data, tested on the same
    # most-recent period none of them has seen.
    def exam_auc(make, fc):
        m = make().fit(train[fc], y(train))
        return roc_auc_score(y(test), m.predict_proba(test[fc])[:, 1]), m

    ch_fc = features_for(challenger_cfg, cols)
    ch_auc, ch_model = exam_auc(lambda: make_model(challenger_cfg), ch_fc)
    champ_auc, champ_model, champ_fc, champ_cfg = None, None, None, None
    if old_bundle and old_bundle.get("demo", False) == args.demo:  # only compare like with like
        champ_cfg = old_bundle.get("config")
        champ_fc = [c for c in old_bundle["columns"] if c in labeled.columns]
        make_champ = ((lambda: make_model(champ_cfg)) if champ_cfg
                      else (lambda: clone(old_bundle["model"])))
        try:
            champ_auc, champ_model = exam_auc(make_champ, champ_fc)
        except Exception as e:
            print(f"  couldn't re-test the old champion: {e}")
    if champ_auc is None or ch_auc > champ_auc + MIN_IMPROVEMENT:
        winner, best_cfg, best_fc, model = "challenger", challenger_cfg, ch_fc, ch_model
        make_best = lambda: make_model(challenger_cfg)
    else:
        winner, best_cfg, best_fc, model = "champion", champ_cfg, champ_fc, champ_model
        make_best = make_champ
    best_auc = ch_auc if winner == "challenger" else champ_auc
    print(f"\nExam on unseen recent data:  champion "
          f"{'(none yet)' if champ_auc is None else f'{champ_auc:.4f}'}  vs  challenger {ch_auc:.4f}"
          f"  ->  {'NEW MODEL WINS' if winner == 'challenger' else 'champion stays'}")
    best_name = describe(best_cfg) if best_cfg else "previous model"
    cols = best_fc

    p = model.predict_proba(test[cols])[:, 1]
    yt = y(test)
    base_rate = yt.mean()
    exam = {
        "model": best_name,
        "winner": winner,
        "champion_auc": None if champ_auc is None else round(float(champ_auc), 4),
        "challenger_auc": round(float(ch_auc), 4),
        "test_period": f"{pd.Timestamp(test_start).date()} to {pd.Timestamp(dates[-1]).date()}",
        "test_rows": int(len(test)),
        "accuracy": round(accuracy_score(yt, p > 0.5) * 100, 1),
        "accuracy_if_you_always_guessed_the_majority": round(max(base_rate, 1 - base_rate) * 100, 1),
        "auc": round(float(best_auc), 4),
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

    # Retrain the winner on everything so the live model knows the most recent market.
    final_model = make_best().fit(labeled_final[cols], y(labeled_final))

    history = prev.get("history", []) + [{
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "auc": exam["auc"], "winner": winner, "model": best_name,
        "coin_days": int(len(labeled))}]
    history = history[-200:]

    # If the model couldn't beat a coin flip on the exam, the analyst won't listen to it.
    model_trusted = exam["auc"] >= MIN_AUC
    exam["model_trusted"] = model_trusted
    joblib.dump({"model": final_model, "columns": cols, "strategy_weights": weights,
                 "model_trusted": model_trusted, "auc": exam["auc"], "config": best_cfg,
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
        "history": history,
        "leaderboard": leaderboard,
        "call_review": review,
    }
    (core.ROOT / "training_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== FINAL EXAM ({exam['test_period']}, never seen during training) ===")
    print(f"Model in use: {best_name}  ({'new this run' if winner == 'challenger' else 'kept from before'})")
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
    if len(history) > 1:
        aucs = [h["auc"] for h in history[-10:]]
        print(f"\nProgress (exam AUC, last {len(aucs)} trainings): " + "  ".join(f"{a:.3f}" for a in aucs))
        print("Note: each run is examined on the newest data, so scores also move with the market.")
    print_review(review)
    if review and review["reviewed"] > (prev.get("call_review") or {}).get("reviewed", 0):
        notify_review(review)
    print("\nSaved model.joblib and training_report.json")
    if args.demo:
        print("\n!! DEMO MODE: trained on fake prices. These numbers say nothing about real crypto.")


if __name__ == "__main__":
    main()
