"""Shared pieces: data download, trading strategies, features, Monte Carlo, notifications."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CG_KEY = os.getenv("COINGECKO_API_KEY", "")
CG_HEADERS = {"x-cg-demo-api-key": CG_KEY} if CG_KEY else {}
HORIZON = 7  # the model predicts the next 7 days

# Coins that are pegged to $1 or wrap another coin - nothing to trade, they'd pollute training.
SKIP_SYMBOLS = {
    "usdt", "usdc", "dai", "busd", "tusd", "usdd", "fdusd", "usde", "pyusd", "usdp", "gusd",
    "frax", "lusd", "eurc", "eurt", "usd0", "usds", "rlusd", "susde", "wbtc", "weth", "steth",
    "wsteth", "cbbtc", "cbeth", "reth", "weeth", "wbeth", "bsc-usd", "xaut", "paxg",
}


# ------------------------------------------------------------------- data ---
def _get(url, params):
    for attempt in range(6):
        r = requests.get(url, params=params, headers=CG_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("CoinGecko rate limit - try again later or add COINGECKO_API_KEY")


def top_coins(n: int) -> list[dict]:
    """The top-n coins by market cap, minus stablecoins and wrapped tokens."""
    coins, page = [], 1
    while len(coins) < n:
        batch = _get("https://api.coingecko.com/api/v3/coins/markets",
                     {"vs_currency": "usd", "order": "market_cap_desc",
                      "per_page": 250, "page": page})
        if not batch:
            break
        coins += [c for c in batch if c["symbol"].lower() not in SKIP_SYMBOLS]
        page += 1
        time.sleep(3)
    return coins[:n]


def history(coin_id: str, use_cache=True, max_age_h=20) -> pd.DataFrame:
    """365 days of daily close prices + volume for one coin (cached on disk)."""
    DATA_DIR.mkdir(exist_ok=True)
    f = DATA_DIR / f"{coin_id}.json"
    if use_cache and f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600:
        raw = json.loads(f.read_text())
    else:
        raw = _get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                   {"vs_currency": "usd", "days": 365, "interval": "daily"})
        f.write_text(json.dumps(raw))
    df = pd.DataFrame({
        "date": pd.to_datetime([p[0] for p in raw["prices"]], unit="ms").normalize(),
        "close": [p[1] for p in raw["prices"]],
        "volume": [v[1] for v in raw["total_volumes"]],
    })
    return df.drop_duplicates("date", keep="last").set_index("date")


# Binance's public market-data server: free, no key, high limits, and not blocked
# for US servers like GitHub's (the main api.binance.com is).
BINANCE = "https://data-api.binance.vision/api/v3"
BINANCE_SKIP = SKIP_SYMBOLS | {"eur", "try", "brl", "aeur", "xusd", "usd1", "bfusd", "euri",
                               "ust", "susd", "idrt", "uah", "bidr", "zar", "ngn", "rub"}


def _binance_get(path, params=None):
    for attempt in range(5):
        r = requests.get(f"{BINANCE}/{path}", params=params, timeout=30)
        if r.status_code in (418, 429):
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Binance rate limit")


def _is_junk(base: str) -> bool:
    """Stablecoins, fiat, and old leveraged tokens (BTCUP, ETHBEAR...) - not real coins."""
    b = base.upper()
    return (base.lower() in BINANCE_SKIP or b.endswith(("DOWN", "BULL", "BEAR"))
            or (b.endswith("UP") and len(b) > 4))


def binance_coins(include_delisted=False) -> list[str]:
    """Every coin trading against USDT on Binance (junk removed). For training we also
    include DELISTED coins - leaving out the ones that died would make the past look
    rosier than it was (survivorship bias)."""
    info = _binance_get("exchangeInfo", {"permissions": "SPOT"})
    return sorted({s["baseAsset"] for s in info["symbols"]
                   if s["quoteAsset"] == "USDT" and not _is_junk(s["baseAsset"])
                   and (s["status"] == "TRADING" or include_delisted)})


def binance_listing_date(symbol: str):
    """The day SYMBOL/USDT first traded on Binance (cached forever - it never changes)."""
    f = DATA_DIR / "binance_listing.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    if symbol not in cache:
        rows = _binance_get("klines", {"symbol": f"{symbol}USDT", "interval": "1d",
                                       "startTime": 0, "limit": 1})
        cache[symbol] = int(rows[0][0]) if rows else None
        DATA_DIR.mkdir(exist_ok=True)
        f.write_text(json.dumps(cache))
    return pd.to_datetime(cache[symbol], unit="ms") if cache[symbol] else None


def binance_history(symbol: str, days=1000, max_age_h=0) -> pd.DataFrame:
    """Up to ~2.7 years of daily candles for SYMBOL/USDT: close, high, low, USD volume,
    number of trades and aggressive-buyer volume. Free and unlimited."""
    f = DATA_DIR / "binance" / f"{symbol}.json"
    if max_age_h and f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600:
        rows = json.loads(f.read_text())
    else:
        rows = _binance_get("klines", {"symbol": f"{symbol}USDT", "interval": "1d", "limit": days})
        if max_age_h:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rows))
    df = pd.DataFrame({
        "close": [float(k[4]) for k in rows],
        "high": [float(k[2]) for k in rows],
        "low": [float(k[3]) for k in rows],
        "volume": [float(k[7]) for k in rows],
        "trades": [float(k[8]) for k in rows],
        "taker_buy": [float(k[10]) for k in rows],
    }, index=pd.to_datetime([k[0] for k in rows], unit="ms").rename("date"))
    try:
        df.attrs["listed"] = binance_listing_date(symbol)
    except Exception:
        pass
    return df


# Market-wide sentiment: Crypto Fear & Greed Index (alternative.me, free, daily since 2018).
OFFLINE = False  # set by --demo so nothing touches the internet
_FNG = {}


def fear_greed() -> pd.Series | None:
    if OFFLINE:
        return None
    if "s" not in _FNG:
        try:
            r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"},
                             timeout=30)
            r.raise_for_status()
            d = r.json()["data"]
            _FNG["s"] = pd.Series([float(x["value"]) for x in d],
                                  index=pd.to_datetime([int(x["timestamp"]) for x in d], unit="s")
                                  .normalize()).sort_index()
        except Exception as e:
            print(f"  Fear & Greed index unavailable ({type(e).__name__})")
            _FNG["s"] = None
    return _FNG["s"]


def fake_history(seed: int, days=365) -> pd.DataFrame:
    """Made-up price history for --demo mode. Regime-switching random walk with a
    little momentum baked in so there's *something* for the model to find."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.02, 0.09)
    drift, rets, prev = 0.0, [], 0.0
    for _ in range(days):
        if rng.random() < 0.03:
            drift = rng.normal(0, vol / 4)
        r = drift + 0.08 * prev + rng.standard_t(4) * vol / 1.4
        rets.append(r)
        prev = r
    close = 10 ** rng.uniform(-3, 4) * np.exp(np.cumsum(rets))
    volume = close * 10 ** rng.uniform(5, 9) * np.exp(rng.normal(0, 0.4, days) + np.abs(rets) * 8)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rets = np.array(rets)
    spread = np.abs(rng.normal(0, vol, days)) + np.abs(rets)
    df = pd.DataFrame({"close": close, "volume": volume,
                       "high": close * (1 + spread * rng.uniform(0.2, 1, days)),
                       "low": close * (1 - spread * rng.uniform(0.2, 1, days)),
                       "trades": volume / close * rng.uniform(0.5, 2) / 50,
                       "taker_buy": volume * np.clip(0.5 + rets * 2 + rng.normal(0, 0.03, days), 0.05, 0.95)},
                      index=idx)
    df.attrs["listed"] = idx[0] - pd.Timedelta(days=int(rng.integers(0, 1500)))
    return df


# ------------------------------------------------------------- strategies ---
# Each strategy is a well-known trading rule. It looks only at the past and says
# +1 (buy), -1 (sell) or 0 (no opinion) for every day.
STRATEGIES = {
    "trend_following": "EMA20 above EMA50 = buy, below = sell (classic trend following)",
    "momentum": "Up over both 7 and 30 days = buy, down over both = sell",
    "rsi_reversion": "RSI below 30 = oversold buy, above 70 = overbought sell",
    "macd": "MACD histogram above zero = buy, below = sell",
    "bollinger_reversion": "Price 2 std devs below its 20-day mean = buy, above = sell",
    "turtle_breakout": "New 20-day high = buy, new 20-day low = sell (Turtle traders)",
    "volume_breakout": "Big move on 1.5x normal volume = follow the move",
    "dip_in_uptrend": "Above 50-day average but RSI < 40 = buy the dip",
}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(df: pd.DataFrame, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Indicators + strategy signals for every day. Only uses data up to that day."""
    c, v = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    f["ret_1d"] = c.pct_change(1)
    f["ret_7d"] = c.pct_change(7)
    f["ret_30d"] = c.pct_change(30)
    f["rsi"] = rsi(c)
    e12, e20, e26, e50 = ema(c, 12), ema(c, 20), ema(c, 26), ema(c, 50)
    macd = e12 - e26
    f["macd_hist"] = (macd - ema(macd, 9)) / c
    f["ema20_vs_50"] = e20 / e50 - 1
    f["price_vs_ema50"] = c / e50 - 1
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_z"] = (c - m20) / s20
    hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
    f["donchian_pos"] = (c - lo20) / (hi20 - lo20)
    f["vol_30d"] = lr.rolling(30).std()
    f["vol_ratio"] = lr.rolling(7).std() / f["vol_30d"]
    f["volume_ratio"] = v.rolling(7).mean() / v.rolling(30).mean()
    f["drawdown_90d"] = c / c.rolling(90, min_periods=30).max() - 1
    if btc is not None:  # what the overall market is doing matters a lot in crypto
        b = btc["close"].reindex(df.index).ffill()
        f["btc_ret_30d"] = b.pct_change(30)
        f["btc_trend"] = ema(b, 20) / ema(b, 50) - 1
        f["vs_btc_30d"] = f["ret_30d"] - f["btc_ret_30d"]  # beating or lagging the market?

    # --- extra data (NaN when a source doesn't have it; the model copes) ---
    nan = pd.Series(np.nan, index=df.index)
    f["ret_90d"] = c.pct_change(90)
    f["drawdown_365d"] = c / c.rolling(365, min_periods=60).max() - 1
    if {"high", "low"} <= set(df.columns):
        h, l = df["high"], df["low"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        f["atr_pct"] = tr.rolling(14).mean() / c                      # true daily volatility
        f["close_in_range"] = ((c - l) / (h - l).replace(0, np.nan)).rolling(3).mean()  # closing strong/weak
    else:
        f["atr_pct"] = f["close_in_range"] = nan
    if "taker_buy" in df.columns:
        tb = df["taker_buy"] / df["volume"].replace(0, np.nan)          # aggressive buyers' share
        f["taker_buy_7d"] = tb.rolling(7).mean()
        f["taker_buy_shift"] = f["taker_buy_7d"] - tb.rolling(30).mean()
    else:
        f["taker_buy_7d"] = f["taker_buy_shift"] = nan
    f["trades_ratio"] = (df["trades"].rolling(7).mean() / df["trades"].rolling(30).mean()
                         if "trades" in df.columns else nan)
    listed = df.attrs.get("listed")
    f["coin_age_days"] = (pd.Series(np.clip((df.index - listed).days, 0, 2000), index=df.index,
                                    dtype=float) if listed is not None else nan)
    fng = fear_greed()
    if fng is not None:
        g = fng.reindex(df.index.normalize()).ffill()
        g.index = df.index
        f["fear_greed"] = g
        f["fear_greed_7d_change"] = g - g.shift(7)
    else:
        f["fear_greed"] = f["fear_greed_7d_change"] = nan

    s = pd.DataFrame(index=df.index)
    s["trend_following"] = np.sign(f["ema20_vs_50"])
    s["momentum"] = np.where((f["ret_7d"] > 0) & (f["ret_30d"] > 0), 1,
                             np.where((f["ret_7d"] < 0) & (f["ret_30d"] < 0), -1, 0))
    s["rsi_reversion"] = np.where(f["rsi"] < 30, 1, np.where(f["rsi"] > 70, -1, 0))
    s["macd"] = np.sign(f["macd_hist"])
    s["bollinger_reversion"] = np.where(f["boll_z"] < -2, 1, np.where(f["boll_z"] > 2, -1, 0))
    s["turtle_breakout"] = np.where(c >= hi20.shift(1), 1, np.where(c <= lo20.shift(1), -1, 0))
    big = f["volume_ratio"] > 1.5
    s["volume_breakout"] = np.where(big & (f["ret_7d"] > 0.05), 1,
                                    np.where(big & (f["ret_7d"] < -0.05), -1, 0))
    s["dip_in_uptrend"] = np.where((f["price_vs_ema50"] > 0) & (f["rsi"] < 40), 1, 0)
    s = s.add_prefix("sig_")
    out = pd.concat([f, s], axis=1)
    out["close"] = c
    return out.iloc[50:]  # drop warm-up days where indicators aren't ready


def add_labels(feat: pd.DataFrame) -> pd.DataFrame:
    """What actually happened next: the return over the following HORIZON days."""
    feat = feat.copy()
    feat["fwd_ret"] = feat["close"].shift(-HORIZON) / feat["close"] - 1
    return feat


def feature_columns(df):
    return [c for c in df.columns if c not in ("close", "fwd_ret", "coin", "date")]


# ------------------------------------------------------------ Monte Carlo ---
def monte_carlo(close: pd.Series, days: int, p_up_7d=None, n_paths=20000, lookback=180,
                block=5, seed=0):
    """Simulate thousands of possible futures by replaying random chunks of this
    coin's own past daily moves (block bootstrap keeps calm/wild streaks together).
    Returns the final % change and the worst drop along the way for every path."""
    rng = np.random.default_rng(seed)
    r = np.log(close).diff().dropna().to_numpy()[-lookback:]
    r = r - r.mean()  # remove past trend so we don't just assume it continues
    n_blocks = -(-days // block)
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    paths = r[starts[..., None] + np.arange(block)].reshape(n_paths, -1)[:, :days]
    if p_up_7d is not None:
        # Tilt the first 7 days so the simulation agrees with the model's prediction
        # (e.g. 62% of paths end the week higher). After day 7 it's a fair coin again.
        k = min(7, days)
        wk = np.sum(r[rng.integers(0, len(r) - block, size=(n_paths, -(-7 // block)))[..., None]
                      + np.arange(block)].reshape(n_paths, -1)[:, :7], axis=1)
        tilt = -np.quantile(wk, 1 - np.clip(p_up_7d, 0.01, 0.99)) / 7
        paths[:, :k] += tilt
    cum = np.cumsum(paths, axis=1)
    final = np.exp(cum[:, -1]) - 1
    worst = np.exp(np.minimum(cum.min(axis=1), 0)) - 1
    return final, worst


def risk_table(close: pd.Series, days: int, p_up_7d=None) -> dict:
    final, worst = monte_carlo(close, days, p_up_7d)
    p = lambda x: round(float(np.mean(x)) * 100, 1)
    return {
        "days": days,
        "simulations": len(final),
        "chance_up": p(final > 0),
        "chance_down": p(final < 0),
        "chance_up_10": p(final > 0.10),
        "chance_up_25": p(final > 0.25),
        "chance_down_10": p(final < -0.10),
        "chance_down_25": p(final < -0.25),
        "chance_down_50": p(final < -0.50),
        "chance_drop_20_at_some_point": p(worst < -0.20),
        "typical_range": [round(float(np.percentile(final, 25)) * 100, 1),
                          round(float(np.percentile(final, 75)) * 100, 1)],
        "worst_5pct": round(float(np.percentile(final, 5)) * 100, 1),   # 95% VaR
        "best_5pct": round(float(np.percentile(final, 95)) * 100, 1),
    }


# ---------------------------------------------------------- notifications ---
def notify(title, message, priority="default", tags=""):
    topic = os.getenv("NTFY_TOPIC", "")
    if not topic:
        print(f"[no NTFY_TOPIC set, would push] {title}\n{message}\n")
        return
    server = os.getenv("NTFY_SERVER", "https://ntfy.sh")
    requests.post(f"{server}/{topic}", data=message.encode(),
                  headers={"Title": title, "Priority": priority, "Tags": tags},
                  timeout=30).raise_for_status()
