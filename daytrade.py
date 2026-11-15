"""Day-trading mode: hourly charts, short trades with a take-profit and a stop-loss.

  python daytrade.py train            # learn from ~4 months of hourly data (+ its own past trades)
  python daytrade.py scan --notify    # look for trades now, check old ones, push signals
  python daytrade.py train --demo     # made-up data, no internet
  python daytrade.py scan --demo

How it works
  * Every possible past trade is simulated: buy at the close of an hour, sell at
    TAKE-PROFIT or STOP-LOSS (whichever is hit first), or after HOLD_HOURS. Targets scale
    with the coin's normal hourly movement (ATR). Fees + slippage are subtracted.
  * The model learns which conditions came before winning trades. Every training run is
    a competition: new model settings must beat the current model on recent hours none
    of them trained on (same champion/challenger system as train.py).
  * It picks the minimum confidence at which trades actually made money after costs.
    If no confidence level made money, it sends NO trade signals.
  * Money management: every signal says how much to buy (risk RISK_PCT of ACCOUNT_SIZE if the
    stop-loss hits), the highest price still worth entering, max MAX_OPEN_TRADES at once, and a
    daily loss limit. A SAFETY SWITCH pauses signals if recent paper trades lose money.
  * Every signal becomes a paper trade in trade_journal.json. Each scan checks how open
    trades ended. When it retrains, its own finished trades count JOURNAL_WEIGHT times
    more than simulated ones - it learns from its own wins and losses.

Long only (buy then sell) - works on a normal spot account. Not financial advice.
"""

import argparse
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import core
import train as T

HOLD_HOURS = 12            # close the trade after this long if neither target is hit
COST = float(os.getenv("TRADE_COST", "0.0025"))  # 0.1% fee each way + ~0.05% slippage
TP_ATR = 1.5               # take-profit = 1.5x the coin's average hourly range
MIN_TP, MAX_TP = 0.008, 0.08
REWARD_RISK = 1.5          # stop-loss = take-profit / 1.5
JOURNAL_WEIGHT = 5         # its own real (paper) trades count 5x in training
TRAIN_HOURS = 3000         # ~4 months of hourly candles per coin
TRAIN_COINS = int(os.getenv("DAYTRADE_COINS", "150"))  # most-traded coins only (tight spreads)
SCAN_COINS = 80
MAX_NEW_TRADES = 3         # per scan

# Money management (set as GitHub variables; these are the defaults)
ACCOUNT = float(os.getenv("ACCOUNT_SIZE") or 1000)                 # your trading account in $
RISK = float(os.getenv("RISK_PCT") or 1) / 100                    # max loss per trade, % of account
MAX_OPEN = int(os.getenv("MAX_OPEN_TRADES") or 3)                 # trades open at the same time
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT_PCT") or 3) / 100  # stop for the day after this
# Safety switch: pause signals if the last KILL_WINDOW paper trades lost money on average
KILL_WINDOW, KILL_MIN_TRADES = 25, 20
MAX_CHASE = 0.25           # don't enter if price already moved 25% of the way to take-profit
MIN_EXAM_TRADES = 100

MODEL_FILE = core.ROOT / "daytrade_model.joblib"
REPORT_FILE = core.ROOT / "daytrade_report.json"
JOURNAL_FILE = core.ROOT / "trade_journal.json"
HOUR_MS = 3_600_000


# ------------------------------------------------------------------- data ---
def _to_df(rows) -> pd.DataFrame:
    return pd.DataFrame({
        "close": [float(k[4]) for k in rows], "high": [float(k[2]) for k in rows],
        "low": [float(k[3]) for k in rows], "volume": [float(k[7]) for k in rows],
        "trades": [float(k[8]) for k in rows], "taker_buy": [float(k[10]) for k in rows],
    }, index=pd.to_datetime([k[0] for k in rows], unit="ms").rename("date"))


def hourly(sym: str, hours: int, drop_open_candle=True) -> pd.DataFrame:
    """Hourly candles for SYM/USDT from Binance (pages back 1000 hours per request)."""
    rows, end = [], None
    while len(rows) < hours:
        params = {"symbol": f"{sym}USDT", "interval": "1h", "limit": min(1000, hours - len(rows))}
        if end:
            params["endTime"] = end
        batch = core._binance_get("klines", params)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
    if drop_open_candle and rows and rows[-1][0] + HOUR_MS > time.time() * 1000:
        rows = rows[:-1]  # the current hour isn't finished yet
    df = _to_df(rows)
    try:
        df.attrs["listed"] = core.binance_listing_date(sym)
    except Exception:
        pass
    return df


def fake_hourly(seed: int, hours: int) -> pd.DataFrame:
    df = core.fake_history(seed, days=hours)
    listed = df.attrs.get("listed")
    # shrink the daily-sized moves to hourly-sized ones
    r = np.log(df["close"]).diff().fillna(0) / 4
    close = df["close"].iloc[0] * np.exp(r.cumsum())
    scale = close / df["close"]
    df = df.mul(scale, axis=0).assign(volume=df["volume"] / 24, trades=df["trades"] / 24,
                                      taker_buy=df["taker_buy"] / 24)
    df["high"] = df[["high", "close"]].max(axis=1)
    df["low"] = df[["low", "close"]].min(axis=1)
    df.index = pd.date_range(end=pd.Timestamp.now().floor("h") - pd.Timedelta(hours=1),
                             periods=hours, freq="h")
    df.attrs["listed"] = listed
    return df


def liquid_coins(n: int) -> list[str]:
    """The n most-traded USDT coins on Binance right now (tight spreads = cheaper day trades)."""
    live = set(core.binance_coins())
    vols = [(float(t["quoteVolume"]), t["symbol"][:-4]) for t in core._binance_get("ticker/24hr")
            if t["symbol"].endswith("USDT") and t["symbol"][:-4] in live]
    return [s for _, s in sorted(vols, reverse=True)[:n]]


# --------------------------------------------------------------- features ---
def features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    f = core.build_features(df, btc)  # same indicators, now on hourly bars
    c, v = df["close"], df["volume"]
    vwap = (c * v).rolling(24).sum() / v.rolling(24).sum()
    extra = pd.DataFrame({
        "hour": df.index.hour, "weekday": df.index.dayofweek,      # time-of-day patterns
        "ret_4h": c.pct_change(4), "ret_24h": c.pct_change(24),
        "vs_vwap_24h": c / vwap - 1,                                 # above/below the day's avg price
        "range_1h": (df["high"] - df["low"]) / c,
    }, index=df.index).reindex(f.index)
    return pd.concat([f, extra], axis=1)


def targets(f: pd.DataFrame):
    tp = (TP_ATR * f["atr_pct"]).clip(MIN_TP, MAX_TP).fillna(MIN_TP)
    return tp, tp / REWARD_RISK


def simulate_trades(df: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    """For every hour: buy at its close, then whichever comes first - take-profit,
    stop-loss, or HOLD_HOURS. Returns the result after costs (NaN when the future
    isn't known yet). If both targets are hit in the same hour, it counts as a loss."""
    d = df.reindex(f.index)
    c, h, l = d["close"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
    tp, sl = (x.to_numpy() for x in targets(f))
    n = len(c)
    first_tp = np.full(n, np.inf)
    first_sl = np.full(n, np.inf)
    for k in range(1, HOLD_HOURS + 1):
        hk = np.full(n, np.nan); hk[:n - k] = h[k:]
        lk = np.full(n, np.nan); lk[:n - k] = l[k:]
        first_tp = np.where(np.isinf(first_tp) & (hk >= c * (1 + tp)), k, first_tp)
        first_sl = np.where(np.isinf(first_sl) & (lk <= c * (1 - sl)), k, first_sl)
    timeout = np.full(n, np.nan)
    timeout[:n - HOLD_HOURS] = c[HOLD_HOURS:] / c[:n - HOLD_HOURS] - 1
    gross = np.where(np.isfinite(first_sl) & (first_sl <= first_tp), -sl,
                     np.where(np.isfinite(first_tp), tp, timeout))
    gross[n - HOLD_HOURS:] = np.nan  # outcome not known yet
    return pd.Series(gross - COST, index=f.index)


# ---------------------------------------------------------------- journal ---
def load_journal():
    return json.loads(JOURNAL_FILE.read_text()) if JOURNAL_FILE.exists() else {"trades": []}


def position_size(sl_pct: float) -> float:
    """$ to buy so that hitting the stop-loss loses RISK of the account (capped so
    MAX_OPEN trades together never exceed the account)."""
    return round(min(ACCOUNT * RISK / max(sl_pct + COST, 1e-6), ACCOUNT / MAX_OPEN), 2)


def pnl(t) -> float:
    """$ result of a closed paper trade."""
    return t["net"] * t.get("size", ACCOUNT / MAX_OPEN)


def journal_stats(journal, days=30, btc_now=None):
    cutoff = time.time() * 1000 - days * 24 * HOUR_MS
    closed = [t for t in journal["trades"] if t["status"] != "open" and t["opened"] >= cutoff]
    if not closed:
        return None
    nets = [t["net"] for t in closed]
    out = {"trades": len(closed), "win_rate": round(100 * np.mean([n > 0 for n in nets]), 1),
           "avg_net_pct": round(100 * float(np.mean(nets)), 2),
           "account_pct": round(100 * sum(pnl(t) for t in closed) / ACCOUNT, 2), "days": days}
    first = min((t for t in closed if t.get("btc_entry")), key=lambda t: t["opened"], default=None)
    if first and btc_now:
        out["btc_pct"] = round(100 * (btc_now / first["btc_entry"] - 1), 2)  # just holding Bitcoin
    return out


def track_line(st) -> str:
    if not st:
        return "No finished paper trades yet."
    line = (f"Last {st['days']} days: {st['trades']} trades, {st['win_rate']}% won, "
            f"{st['avg_net_pct']:+}% avg after costs | paper account {st['account_pct']:+}%")
    if "btc_pct" in st:
        beat = "BEATING" if st["account_pct"] > st["btc_pct"] else "behind"
        line += f" vs holding BTC {st['btc_pct']:+}% ({beat})"
    return line


def safety_check(journal, bundle, now_ms):
    """Safety switch. Pauses new signals when recent paper trades lose money on average;
    resumes only after a NEW training run finds an edge again. Returns (ok, message_if_changed)."""
    st = journal.setdefault("safety", {"paused": False, "resumed_at": 0})
    if st["paused"]:
        if bundle.get("trained_at", 0) > st["since"] and bundle.get("threshold") is not None:
            st.update(paused=False, resumed_at=now_ms)
            return True, ("Safety switch OFF: a new training found an edge again, so signals are back on. "
                          "Its record starts fresh from now.")
        return False, None
    recent = [t for t in journal["trades"] if t["status"] != "open" and t["opened"] >= st["resumed_at"]]
    recent = sorted(recent, key=lambda t: t["opened"])[-KILL_WINDOW:]
    if len(recent) >= KILL_MIN_TRADES and np.mean([t["net"] for t in recent]) < 0:
        avg = 100 * float(np.mean([t["net"] for t in recent]))
        st.update(paused=True, since=now_ms, reason=f"last {len(recent)} trades averaged {avg:+.2f}%")
        return False, (f"Safety switch ON: my last {len(recent)} paper trades lost money on average "
                       f"({avg:+.2f}% per trade). I'm pausing buy signals until a new training "
                       "finds an edge again. Open trades are still tracked.")
    return True, None


def day_loss_hit(journal, now_ms) -> bool:
    day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).date()
    today = [t for t in journal["trades"] if t["status"] != "open" and t.get("closed")
             and datetime.fromtimestamp(t["closed"] / 1000, timezone.utc).date() == day]
    return sum(pnl(t) for t in today) <= -DAILY_LOSS_LIMIT * ACCOUNT


def resolve_open_trades(journal, demo):
    """Check how open paper trades ended, using the hours since they were opened."""
    now_ms = time.time() * 1000
    for t in journal["trades"]:
        if t["status"] != "open":
            continue
        try:
            if demo:
                rows = fake_hourly(zlib.crc32(t["coin"].encode()) + int(t["opened"] // HOUR_MS), HOLD_HOURS + 1)
                bars = list(zip(rows["high"] / rows["close"].iloc[0] * t["entry"],
                                rows["low"] / rows["close"].iloc[0] * t["entry"],
                                rows["close"] / rows["close"].iloc[0] * t["entry"],
                                [t["opened"] + (i + 1) * HOUR_MS for i in range(len(rows))]))
                bars = [b for b in bars if b[3] + HOUR_MS <= now_ms]  # only hours that have passed
            else:
                k = core._binance_get("klines", {"symbol": f"{t['coin']}USDT", "interval": "1h",
                                                 "startTime": int(t["opened"]) + HOUR_MS,
                                                 "limit": HOLD_HOURS})
                bars = [(float(r[2]), float(r[3]), float(r[4]), r[0]) for r in k
                        if r[0] + HOUR_MS <= now_ms]  # finished hours only
        except Exception as e:
            print(f"  couldn't check {t['coin']} trade: {e}", file=sys.stderr)
            continue
        for i, (hi, lo, cl, _) in enumerate(bars[:HOLD_HOURS], 1):
            if lo <= t["sl"]:
                t.update(status="lost", exit=t["sl"]); break
            if hi >= t["tp"]:
                t.update(status="won", exit=t["tp"]); break
            if i == HOLD_HOURS:
                t.update(status="timeout", exit=cl)
        if t["status"] != "open":
            t["net"] = t["exit"] / t["entry"] - 1 - COST
            t["closed"] = int(now_ms)


def journal_rows(journal, columns):
    """Its own finished trades as extra training examples."""
    rows = [t for t in journal["trades"] if t["status"] != "open" and t.get("features")]
    if not rows:
        return pd.DataFrame()
    X = pd.DataFrame([{c: t["features"].get(c) for c in columns} for t in rows], dtype=float)
    X["net"] = [t["net"] for t in rows]
    X["date"] = pd.to_datetime([t["opened"] for t in rows], unit="ms")
    return X


# ---------------------------------------------------------------- training ---
def fit_weighted(model, X, y, w):
    if isinstance(model, Pipeline):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def pick_threshold(p, net):
    """The confidence level where taking every trade above it made the most money after
    costs. Tries the model's top 50%, 40%, ... down to its top 0.5% most confident calls."""
    best = None
    for q in np.r_[np.arange(0.50, 0.99, 0.02), 0.99, 0.995]:
        t = float(np.quantile(p, q))
        m = p >= t
        if m.sum() < MIN_EXAM_TRADES:
            break
        avg = float(net[m].mean())
        if avg > 0 and (best is None or avg * np.sqrt(m.sum()) > best[1] * np.sqrt(best[2])):
            best = (round(t, 4), avg, int(m.sum()))
    return best


def cmd_train(args):
    rng = np.random.default_rng()
    print("Downloading hourly candles..." if not args.demo else "DEMO: fake hourly data")
    coins = ([f"DEMO{i}" for i in range(args.coins)] if args.demo else liquid_coins(args.coins))
    load = ((lambda s: fake_hourly(zlib.crc32(s.encode()), TRAIN_HOURS)) if args.demo
            else (lambda s: hourly(s, TRAIN_HOURS)))
    btc = fake_hourly(0, TRAIN_HOURS) if args.demo else hourly("BTC", TRAIN_HOURS)
    frames = []
    for i, s in enumerate(coins):
        try:
            df = btc if s == "BTC" else load(s)
            if len(df) < 500:
                continue
            f = features(df, btc)
            f["net"] = simulate_trades(df, f)
            f["coin"] = s
            frames.append(f)
        except Exception as e:
            print(f"  skip {s}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(coins)} coins")
    data = pd.concat(frames).rename_axis("date").reset_index().replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["net"])
    cols = [c for c in core.feature_columns(data) if c != "net" and data[c].notna().mean() > 0.05]
    y = lambda d: (d["net"] > 0).astype(int)
    print(f"\nSimulated past trades: {len(data):,} from {data['coin'].nunique()} coins "
          f"({100 * y(data).mean():.1f}% were winners after costs)")

    journal = load_journal()
    resolve_open_trades(journal, args.demo)  # check how its latest trades ended before learning
    jr = journal_rows(journal, cols)
    if len(jr):
        print(f"Its own finished paper trades: {len(jr)} (each counts {JOURNAL_WEIGHT}x)")

    times = np.sort(data["date"].unique())
    gap = pd.Timedelta(hours=HOLD_HOURS)
    test_start = times[int(len(times) * 0.8)]
    trn = data[data["date"] < test_start - gap]
    test = data[data["date"] >= test_start]
    val_start = np.sort(trn["date"].unique())[int(trn["date"].nunique() * 0.8)]
    fit, val = trn[trn["date"] < val_start - gap], trn[trn["date"] >= val_start]

    def with_journal(d, before):
        j = jr[jr["date"] < before] if len(jr) else jr
        X = pd.concat([d[cols + ["net"]], j[cols + ["net"]]]) if len(j) else d[cols + ["net"]]
        w = np.r_[np.ones(len(d)), np.full(len(j), JOURNAL_WEIGHT)] if len(j) else np.ones(len(d))
        return X, w

    # --- champion vs challengers (same idea as train.py) ---
    prev = json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else {}
    old = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    cands = [c["config"] for c in prev.get("leaderboard", [])[:T.KEEP_BEST]] or list(T.STARTERS)
    cands += [T.random_config(rng) for _ in range(args.explore)]
    sfit = fit.sample(min(len(fit), T.SEARCH_ROWS), random_state=0)
    Xs, ws = with_journal(sfit, val_start - gap)
    print(f"\nSearching {len(cands)} model settings (validation AUC, 0.50 = coin flip):")
    scored = []
    for cfg in cands:
        fc = T.features_for(cfg, cols)
        try:
            m = fit_weighted(T.make_model(cfg), Xs[fc], y(Xs), ws)
            auc = roc_auc_score(y(val), m.predict_proba(val[fc])[:, 1])
        except Exception as e:
            print(f"  failed {T.describe(cfg)}: {e}")
            continue
        print(f"  {auc:.4f}  {T.describe(cfg)}")
        scored.append((auc, cfg))
    scored.sort(key=lambda t: -t[0])
    leaderboard, seen = [], set()
    for auc, cfg in scored:
        k = json.dumps(cfg, sort_keys=True)
        if k not in seen:
            seen.add(k)
            leaderboard.append({"config": cfg, "val_auc": round(float(auc), 4)})

    Xt, wt = with_journal(trn, test_start - gap)

    def exam(make, fc):
        m = fit_weighted(make(), Xt[fc], y(Xt), wt)
        p = m.predict_proba(test[fc])[:, 1]
        return roc_auc_score(y(test), p), p

    ch_cfg = scored[0][1]
    ch_fc = T.features_for(ch_cfg, cols)
    ch_auc, ch_p = exam(lambda: T.make_model(ch_cfg), ch_fc)
    champ_auc = None
    if old and old.get("demo", False) == args.demo:
        champ_cfg = old.get("config")
        champ_fc = [c for c in old["columns"] if c in cols]
        make_champ = (lambda: T.make_model(champ_cfg)) if champ_cfg else (lambda: clone(old["model"]))
        try:
            champ_auc, champ_p = exam(make_champ, champ_fc)
        except Exception as e:
            print(f"  couldn't re-test the old model: {e}")
    if champ_auc is None or ch_auc > champ_auc + T.MIN_IMPROVEMENT:
        winner, cfg, fc, auc, p, make = "challenger", ch_cfg, ch_fc, ch_auc, ch_p, (lambda: T.make_model(ch_cfg))
    else:
        winner, cfg, fc, auc, p, make = "champion", champ_cfg, champ_fc, champ_auc, champ_p, make_champ
    print(f"\nExam on the most recent {len(test):,} hours x coins (never trained on): champion "
          f"{'(none yet)' if champ_auc is None else f'{champ_auc:.4f}'} vs challenger {ch_auc:.4f}"
          f" -> {'NEW MODEL WINS' if winner == 'challenger' else 'champion stays'}")

    net = test["net"].to_numpy()
    th = pick_threshold(p, net)
    take_all = float(net.mean())
    if th:
        m = p >= th[0]
        stats = {"threshold": th[0], "trades": int(m.sum()),
                 "win_rate": round(100 * float((net[m] > 0).mean()), 1),
                 "avg_net_pct": round(100 * th[1], 3)}
        print(f"Signals when confidence >= {th[0]:.0%}: {stats['trades']:,} exam trades, "
              f"{stats['win_rate']}% winners, {stats['avg_net_pct']:+}% average per trade after costs")
    else:
        stats = None
        print("!! No confidence level made money after costs on the exam -> NO day-trade signals "
              "will be sent until a future training finds an edge.")
    print(f"(Taking every trade blindly averaged {100 * take_all:+.3f}% after costs.)")

    Xall, wall = with_journal(data, data["date"].max() + gap)
    final = fit_weighted(make(), Xall[fc], y(Xall), wall)
    history = (prev.get("history", []) + [{
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "auc": round(float(auc), 4),
        "winner": winner, "edge": bool(th), "avg_net_pct": stats["avg_net_pct"] if stats else None,
        "journal_trades": int(len(jr))}])[-200:]
    joblib.dump({"model": final, "columns": fc, "config": cfg, "auc": float(auc),
                 "threshold": th[0] if th else None, "exam_stats": stats, "demo": args.demo,
                 "trained_at": int(time.time() * 1000)},
                MODEL_FILE)
    REPORT_FILE.write_text(json.dumps({
        "trained_at": history[-1]["date"], "demo_data": args.demo, "coins": int(data["coin"].nunique()),
        "simulated_trades": int(len(data)), "own_paper_trades_used": int(len(jr)),
        "hold_hours": HOLD_HOURS, "cost_per_trade_pct": COST * 100,
        "exam": {"auc": round(float(auc), 4), "winner": winner, "model": T.describe(cfg) if cfg else "previous",
                 "signals": stats, "take_every_trade_avg_net_pct": round(100 * take_all, 3)},
        "own_trades_last_30d": journal_stats(journal),
        "inputs": fc, "history": history, "leaderboard": leaderboard[:T.LEADERBOARD_SIZE],
    }, indent=2))
    if len(history) > 1:
        print("Progress (exam AUC): " + "  ".join(f"{h['auc']:.3f}" for h in history[-10:]))
    print("Saved daytrade_model.joblib and daytrade_report.json")
    if args.demo:
        print("!! DEMO MODE - fake data, numbers mean nothing.")


# ------------------------------------------------------------------- scan ---
def fmt(p):
    return f"${p:,.4f}" if p < 10 else f"${p:,.2f}"


def cmd_scan(args):
    if not MODEL_FILE.exists():
        sys.exit("No day-trade model yet - run: python daytrade.py train")
    bundle = joblib.load(MODEL_FILE)
    journal = load_journal()
    resolve_open_trades(journal, args.demo)

    coins = [f"DEMO{i}" for i in range(SCAN_COINS)] if args.demo else liquid_coins(SCAN_COINS)
    btc = fake_hourly(0, 400) if args.demo else hourly("BTC", 400)
    now_ms = int(time.time() * 1000)
    rows = []
    for s in coins:
        try:
            df = (btc if s == "BTC" else
                  fake_hourly(zlib.crc32(s.encode()) + now_ms // HOUR_MS, 400) if args.demo else hourly(s, 400))
            f = features(df, btc).replace([np.inf, -np.inf], np.nan)
            last = f.iloc[[-1]]
            p = float(bundle["model"].predict_proba(last[bundle["columns"]])[:, 1][0])
            tp, sl = (float(x.iloc[-1]) for x in targets(f))
            rows.append({"coin": s, "p": p, "price": float(df["close"].iloc[-1]), "tp_pct": tp,
                         "sl_pct": sl, "features": {c: (None if pd.isna(v) else float(v))
                                                    for c, v in last.iloc[0].items() if c != "close"}})
        except Exception as e:
            print(f"  skip {s}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: -r["p"])

    th = bundle.get("threshold")
    print(f"\nDay-trade scan of {len(rows)} most-traded coins | hold max {HOLD_HOURS}h | "
          f"signal threshold: {f'{th:.0%}' if th is not None else 'NONE (no proven edge)'}")
    print(f"{'coin':10}{'price':>14}{'win chance':>12}{'take-profit':>13}{'stop-loss':>11}")
    for r in rows[:12]:
        print(f"{r['coin']:10}{fmt(r['price']):>14}{r['p']:>11.0%}{r['tp_pct']:>+12.1%}{-r['sl_pct']:>+11.1%}")

    btc_now = float(btc["close"].iloc[-1])
    ok, safety_msg = safety_check(journal, bundle, now_ms)
    open_trades = [t for t in journal["trades"] if t["status"] == "open"]
    open_coins = {t["coin"] for t in open_trades}
    blocked = None
    if th is None:
        blocked = "no proven edge in the last training"
    elif not ok:
        blocked = f"safety switch is ON ({journal['safety'].get('reason', '')})"
    elif day_loss_hit(journal, now_ms):
        blocked = f"daily loss limit hit (-{DAILY_LOSS_LIMIT:.0%} of account) - back tomorrow"
    elif len(open_trades) >= MAX_OPEN:
        blocked = f"already {len(open_trades)} trades open (max {MAX_OPEN})"
    new = []
    if not blocked:
        room = min(MAX_NEW_TRADES, MAX_OPEN - len(open_trades))
        for r in rows:
            if len(new) >= room:
                break
            if r["p"] >= th and r["coin"] not in open_coins:
                size = position_size(r["sl_pct"])
                t = {"coin": r["coin"], "opened": now_ms, "entry": r["price"],
                     "tp": r["price"] * (1 + r["tp_pct"]), "sl": r["price"] * (1 - r["sl_pct"]),
                     "max_entry": r["price"] * (1 + MAX_CHASE * r["tp_pct"]),
                     "deadline": now_ms + HOLD_HOURS * HOUR_MS, "prob": round(r["p"], 4),
                     "size": size, "btc_entry": btc_now,
                     "status": "open", "features": r["features"]}
                journal["trades"].append(t)
                new.append((r, t))

    st = journal_stats(journal, btc_now=btc_now)
    track = track_line(st)
    print("\n" + track)
    print(f"Money rules: account ${ACCOUNT:,.0f}, risk {RISK:.1%}/trade, max {MAX_OPEN} open, "
          f"daily loss limit {DAILY_LOSS_LIMIT:.0%}. Open now: {len(open_trades)}")
    if safety_msg:
        print(safety_msg)
        if args.notify:
            core.notify("Day-trade safety switch", safety_msg + f"\n\n{track}", priority="high",
                        tags="octagonal_sign" if not ok else "green_circle")
    if blocked:
        print(f"No new signals: {blocked}.")
    for r, t in new:
        risk_usd = t["size"] * (r["sl_pct"] + COST)
        print(f"NEW: BUY {r['coin']} @ {fmt(t['entry'])} (max {fmt(t['max_entry'])})  "
              f"TP {fmt(t['tp'])} ({r['tp_pct']:+.1%})  SL {fmt(t['sl'])} ({-r['sl_pct']:+.1%})  "
              f"size ${t['size']:,.0f} (risk ${risk_usd:,.2f})  win chance {r['p']:.0%}")
        if args.notify:
            core.notify(f"DAY TRADE: BUY {r['coin']} @ {fmt(t['entry'])}",
                        f"Don't buy above: {fmt(t['max_entry'])} (if it's higher, skip it)\n"
                        f"Take-profit: {fmt(t['tp'])} ({r['tp_pct']:+.1%})\n"
                        f"Stop-loss: {fmt(t['sl'])} ({-r['sl_pct']:+.1%})\n"
                        f"Close after {HOLD_HOURS}h if neither hits.\n"
                        f"Size: ${t['size']:,.0f} -> max loss ${risk_usd:,.2f} "
                        f"({RISK:.1%} of ${ACCOUNT:,.0f})\n"
                        f"Win chance {r['p']:.0%} (signals start at {th:.0%})\n\n{track}",
                        priority="high", tags="zap")
    if not new and not blocked:
        print("No new trade signals this hour.")

    # Tell the user when paper trades finish (once each).
    for t in journal["trades"]:
        if t["status"] != "open" and not t.get("reported"):
            t["reported"] = True
            if args.notify:
                core.notify(f"Trade closed: {t['coin']} {t['status'].upper()} {100 * t['net']:+.2f}% "
                            f"(${pnl(t):+,.2f})",
                            f"Entry {fmt(t['entry'])} -> exit {fmt(t['exit'])} (after costs)\n{track}",
                            tags="white_check_mark" if t["net"] > 0 else "x")
    journal["trades"] = [t for t in journal["trades"]
                         if t["opened"] > now_ms - 120 * 24 * HOUR_MS]  # keep ~4 months
    if not args.demo:
        JOURNAL_FILE.write_text(json.dumps(journal))
    print("\nPaper trades only - nothing is bought for you. Day trading loses money for most "
          "people; fees add up fast. Not financial advice.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "scan"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--coins", type=int, default=TRAIN_COINS)
    ap.add_argument("--explore", type=int, default=4)
    args = ap.parse_args()
    core.OFFLINE = args.demo
    cmd_train(args) if args.command == "train" else cmd_scan(args)


if __name__ == "__main__":
    main()
