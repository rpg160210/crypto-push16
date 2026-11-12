"""Hype radar: which coins are trending on social media and search RIGHT NOW.

  python hype.py            # print the hype ranking
  python hype.py --notify   # also push the top 5 to your phone when it changes
  python hype.py --demo     # made-up data, no internet

100% free sources, no keys required:
  * CoinGecko "Trending"  - most-searched coins right now (uses COINGECKO_API_KEY if set)
  * Reddit                - coin mentions in new posts on crypto subreddits
  * Bluesky               - coin mentions in recent crypto posts
  * 4chan /biz/           - coin mentions on the crypto board
  * Binance               - is real money following the hype? (volume spike, price move)

Each coin gets a HYPE SCORE (0-100) and a STAGE:
  EARLY    - buzz is building but the price hasn't run yet
  BUILDING - buzz and price both rising
  LATE     - price already pumped hard; buying now = buying someone else's exit
  FADING   - people are talking about it less than before
  MILD     - a bit of chatter, not real hype

Hype trading is closer to gambling than investing. Most hyped coins dump.
Nothing here has been back-tested (free historical social data doesn't exist),
so the bot saves every reading to hype_history.json to build its own record.
"""

import argparse
import html
import json
import math
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import core

HISTORY_FILE = core.ROOT / "hype_history.json"
UA = {"User-Agent": "crypto-hype-radar/1.0 (open-source hobby project)"}
SUBREDDITS = ["CryptoCurrency", "CryptoMoonShots", "SatoshiStreetBets", "altcoin",
              "CryptoMarkets", "memecoins"]
BLUESKY_QUERIES = ["crypto", "altcoin", "memecoin"]
DAY = 86400

# Uppercase words that are also tickers but almost always mean the normal word.
# (Written as a cashtag, e.g. $ONE, they still count.)
BARE_STOPWORDS = {
    "THE", "AND", "FOR", "ONE", "ALL", "CAN", "NOW", "NEW", "GET", "OUT", "BIG", "TOP", "HOT",
    "WIN", "BUY", "SELL", "PUMP", "DUMP", "MOON", "USD", "USDT", "USDC", "CEO", "ATH", "ETF",
    "NFT", "DEFI", "APE", "GAS", "SUN", "PEOPLE", "MEME", "TRUMP", "REAL", "FUN", "ACE", "JOE",
    "HOOK", "MASK", "SAND", "EDU", "DOGS", "AUDIO", "MAGIC", "GOAT", "HIGH", "LOW", "KEY",
    "DATA", "BOND", "HARD", "TRU", "ALT", "OMG", "WOO", "BAT", "COW", "CAT", "DOG", "PAY",
    "GIGA", "TURBO", "BANANA", "NOT", "BOME", "WIF", "AI", "ID", "OP", "IO", "GPT", "IMO",
    "LOL", "WTF", "FUD", "FOMO", "DCA", "API", "TVL", "CEX", "DEX", "SEC", "FED", "CPI",
}
# Full names people actually write -> ticker.
NAMES = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "dogecoin": "DOGE", "ripple": "XRP",
    "cardano": "ADA", "shiba": "SHIB", "pepe": "PEPE", "chainlink": "LINK", "polkadot": "DOT",
    "avalanche": "AVAX", "litecoin": "LTC", "tron": "TRX", "toncoin": "TON", "polygon": "POL",
    "arbitrum": "ARB", "optimism": "OP", "sui": "SUI", "aptos": "APT", "bonk": "BONK",
    "dogwifhat": "WIF", "floki": "FLOKI", "render": "RENDER", "injective": "INJ",
    "hyperliquid": "HYPE", "uniswap": "UNI", "aave": "AAVE", "stellar": "XLM", "hedera": "HBAR",
    "kaspa": "KAS", "monero": "XMR", "near": "NEAR", "fetch.ai": "FET", "bittensor": "TAO",
}
CASHTAG = re.compile(r"\$([A-Za-z][A-Za-z0-9]{1,9})\b")
BARE = re.compile(r"\b([A-Z][A-Z0-9]{2,9})\b")
WORD = re.compile(r"[a-z][a-z.]+")


# ---------------------------------------------------------------- sources ---
def clean(text):
    return html.unescape(re.sub(r"<[^>]+>", " ", text or ""))


def reddit_posts():
    """Newest posts from crypto subreddits. Tries the JSON API, then the RSS feed
    (Reddit sometimes blocks cloud servers from one but not the other)."""
    texts, cutoff = [], time.time() - DAY
    for sub in SUBREDDITS:
        try:
            r = requests.get(f"https://www.reddit.com/r/{sub}/new.json",
                             params={"limit": 100}, headers=UA, timeout=20)
            r.raise_for_status()
            for c in r.json()["data"]["children"]:
                d = c["data"]
                if d.get("created_utc", 0) >= cutoff:
                    texts.append(f"{d.get('title', '')} {d.get('selftext', '')}")
        except Exception:
            try:
                r = requests.get(f"https://www.reddit.com/r/{sub}/new/.rss",
                                 params={"limit": 100}, headers=UA, timeout=20)
                r.raise_for_status()
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for e in ET.fromstring(r.content).findall("a:entry", ns):
                    when = e.findtext("a:updated", "", ns) or e.findtext("a:published", "", ns)
                    if when and datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp() < cutoff:
                        continue
                    texts.append(f"{e.findtext('a:title', '', ns)} {clean(e.findtext('a:content', '', ns))}")
            except Exception as e:
                print(f"  reddit r/{sub} unavailable ({type(e).__name__})", file=sys.stderr)
        time.sleep(1.5)
    return texts


def bluesky_posts():
    texts, cutoff = [], time.time() - DAY
    for q in BLUESKY_QUERIES:
        try:
            r = requests.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
                             params={"q": q, "limit": 100, "sort": "latest"}, headers=UA, timeout=20)
            r.raise_for_status()
            for p in r.json().get("posts", []):
                rec = p.get("record", {})
                when = rec.get("createdAt", "")
                try:
                    if datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp() < cutoff:
                        continue
                except ValueError:
                    pass
                texts.append(rec.get("text", ""))
        except Exception as e:
            print(f"  bluesky '{q}' unavailable ({type(e).__name__})", file=sys.stderr)
    return texts


def biz_posts():
    """Threads (+ latest replies) on 4chan /biz/. Read-only public API."""
    texts, cutoff = [], time.time() - DAY
    try:
        r = requests.get("https://a.4cdn.org/biz/catalog.json", headers=UA, timeout=20)
        r.raise_for_status()
        for page in r.json():
            for t in page.get("threads", []):
                if t.get("time", 0) >= cutoff:
                    texts.append(f"{t.get('sub', '')} {clean(t.get('com', ''))}")
                for rep in t.get("last_replies", []):
                    if rep.get("time", 0) >= cutoff:
                        texts.append(clean(rep.get("com", "")))
    except Exception as e:
        print(f"  4chan /biz/ unavailable ({type(e).__name__})", file=sys.stderr)
    return texts


def coingecko_trending():
    """{SYMBOL: rank} for CoinGecko's most-searched coins (rank 0 = #1)."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search/trending",
                         headers=core.CG_HEADERS, timeout=20)
        r.raise_for_status()
        out = {}
        for i, c in enumerate(r.json().get("coins", [])):
            sym = c["item"]["symbol"].upper()
            out.setdefault(sym, i)
        return out
    except Exception as e:
        print(f"  CoinGecko trending unavailable ({type(e).__name__})", file=sys.stderr)
        return {}


def count_mentions(texts, symbols):
    """How many posts mention each coin (a post counts once per coin)."""
    counts = {}
    for t in texts:
        found = {m.upper() for m in CASHTAG.findall(t)}
        found |= {m for m in BARE.findall(t) if m not in BARE_STOPWORDS}
        found |= {NAMES[w] for w in WORD.findall(t.lower()) if w in NAMES}
        for s in found & symbols:
            counts[s] = counts.get(s, 0) + 1
    return counts


# ------------------------------------------------------------------ demo ---
def demo_inputs():
    rng = random.Random(7)
    syms = ["BTC", "ETH", "SOL", "DOGE", "PEPE", "WIF", "BONK", "SUI", "TAO", "FET", "XRP",
            "ADA", "HYPE", "FLOKI", "RENDER", "INJ", "ARB", "TRX", "LINK", "NEAR"]
    mentions = {s: int(rng.expovariate(1 / 6)) for s in syms}
    mentions.update({"PEPE": 41, "WIF": 23, "TAO": 17})
    trending = {"PEPE": 0, "TAO": 1, "SUI": 2, "FLOKI": 3, "HYPE": 4, "WIF": 5}
    tickers = {s: {"change_24h": rng.uniform(-8, 12), "volume_24h": 10 ** rng.uniform(6, 9),
                   "price": 10 ** rng.uniform(-5, 4)} for s in syms}
    tickers["PEPE"]["change_24h"] = 31.0
    return set(syms), mentions, trending, tickers, {"reddit": 600, "bluesky": 300, "biz": 400}


def demo_history(sym):
    return core.fake_history(sum(map(ord, sym)), days=120)


# ----------------------------------------------------------------- scoring ---
def binance_tickers():
    rows = core._binance_get("ticker/24hr")
    out = {}
    for t in rows:
        if t["symbol"].endswith("USDT"):
            out[t["symbol"][:-4]] = {"change_24h": float(t["priceChangePercent"]),
                                     "volume_24h": float(t["quoteVolume"]),
                                     "price": float(t["lastPrice"])}
    return out


def score_coin(sym, mentions, base, trend_rank, tick, hist):
    c, v = hist["close"], hist["volume"]
    prev_vol = v.iloc[-31:-1].median() if len(v) > 5 else np.nan
    vol_ratio = tick["volume_24h"] / prev_vol if prev_vol and prev_vol > 0 else 1.0
    change_7d = (c.iloc[-1] / c.iloc[-8] - 1) * 100 if len(c) > 8 else 0.0
    rsi = float(core.rsi(c).iloc[-1]) if len(c) > 20 else 50.0
    growth = (mentions + 1) / (base + 1) if base is not None else None

    pts = {
        "trending": 30 * (1 - trend_rank / 15) if trend_rank is not None else 0,
        "social": min(30, 10 * math.log2(1 + mentions)),
        "growth": min(15, 7.5 * math.log2(growth)) if growth and growth > 1 else 0,
        "volume": min(15, 7.5 * math.log2(vol_ratio)) if vol_ratio > 1 else 0,
        "price": min(10, max(0, tick["change_24h"] / 3)),
    }
    score = round(sum(pts.values()))

    if score < 35:
        stage = "MILD"
    elif change_7d > 60 or rsi > 80 or tick["change_24h"] > 40:
        stage = "LATE"
    elif growth is not None and growth < 0.7 and vol_ratio < 1:
        stage = "FADING"
    elif change_7d < 20 and tick["change_24h"] < 15:
        stage = "EARLY"
    else:
        stage = "BUILDING"
    return {"coin": sym, "hype_score": score, "stage": stage, "mentions_24h": mentions,
            "mention_growth": round(growth, 2) if growth else None,
            "trending_rank": trend_rank + 1 if trend_rank is not None else None,
            "volume_spike": round(vol_ratio, 1), "change_24h": round(tick["change_24h"], 1),
            "change_7d": round(change_7d, 1), "rsi": round(rsi), "price": tick["price"],
            "points": {k: round(v) for k, v in pts.items()}, "close": c}


OPINION = {
    "EARLY": "Buzz is building before the price has really moved. This is the only stage "
             "where hype trades sometimes pay - and most still fizzle. Tiny size, set a stop.",
    "BUILDING": "Hype and price are rising together. You're not early anymore; momentum can carry "
                "it further but the exit door gets crowded fast.",
    "LATE": "It already pumped hard. Buying here usually means buying from the people who got in "
            "early. I'd stay away or wait for a big pullback.",
    "MILD": "Some chatter, nothing special. Not a hype play right now.",
    "FADING": "People are talking about it less than before. Hype that fades rarely comes back "
              "quickly - not one I'd chase.",
}


def fmt_price(p):
    return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--notify", action="store_true", help="push the top 5 when it changes")
    ap.add_argument("--notify-all", action="store_true", help="push the top 5 every run")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else {"runs": []}

    print("Reading social media and trending lists...")
    if args.demo:
        symbols, mentions, trending, tickers, sizes = demo_inputs()
        load = demo_history
    else:
        symbols = set(core.binance_coins())
        tickers = binance_tickers()
        reddit, bsky, biz = reddit_posts(), bluesky_posts(), biz_posts()
        sizes = {"reddit": len(reddit), "bluesky": len(bsky), "biz": len(biz)}
        mentions = count_mentions(reddit + bsky + biz, symbols)
        trending = coingecko_trending()

        def load(sym):
            rows = core._binance_get("klines", {"symbol": f"{sym}USDT", "interval": "1d", "limit": 120})
            return pd.DataFrame({"close": [float(k[4]) for k in rows],
                                 "volume": [float(k[7]) for k in rows]})
    print(f"Posts read in the last 24h: {sizes}  |  CoinGecko trending: {len(trending)} coins")
    if sum(sizes.values()) == 0 and not trending:
        sys.exit("!! No social data could be read this run (all sources blocked or down).")

    # Baseline = average mentions over previous runs in the last 7 days.
    recent = [r for r in history["runs"] if r["time"] > time.time() - 7 * DAY]
    def baseline(sym):
        if len(recent) < 3:
            return None  # not enough history yet
        return sum(r["mentions"].get(sym, 0) for r in recent) / len(recent)

    # Only coins tradable on Binance with real volume, that have *some* buzz.
    candidates = {s for s in symbols if s in tickers and tickers[s]["volume_24h"] > 500_000
                  and (mentions.get(s, 0) >= 2 or s in trending)}
    results = []
    for s in candidates:
        try:
            hist = load(s)
            if len(hist) < 10:
                continue
            if args.demo:  # make the fake 24h numbers match the fake history
                tickers[s]["volume_24h"] = hist["volume"].iloc[-30:].median() * random.uniform(0.6, 3)
                tickers[s]["price"] = hist["close"].iloc[-1]
            results.append(score_coin(s, mentions.get(s, 0), baseline(s), trending.get(s),
                                      tickers[s], hist))
        except Exception as e:
            print(f"  skipped {s}: {e}", file=sys.stderr)
    results.sort(key=lambda r: -r["hype_score"])
    top = results[:args.top]

    print(f"\n{'':4}{'coin':8}{'hype':>5}  {'stage':9}{'mentions':>9}{'vs avg':>8}{'trend#':>7}"
          f"{'vol x':>7}{'24h':>8}{'7d':>8}{'RSI':>5}")
    for i, r in enumerate(top, 1):
        g = f"{r['mention_growth']}x" if r["mention_growth"] else "new"
        vx = f"{r['volume_spike']}x" if r["volume_spike"] < 100 else ">99x"
        t = r["trending_rank"] or "-"
        print(f"{i:>3} {r['coin']:8}{r['hype_score']:>5}  {r['stage']:9}{r['mentions_24h']:>9}"
              f"{g:>8}{t:>7}{vx:>7}{r['change_24h']:>+7}%{r['change_7d']:>+7}%{r['rsi']:>5}")
    if len(recent) < 3:
        print("\n(The 'vs avg' column fills in after a few runs - the bot is still learning "
              "what normal buzz looks like for each coin.)")

    for r in top[:5]:
        k7, k30 = core.risk_table(r["close"], 7), core.risk_table(r["close"], 30)
        r["risk"] = {"7d": k7, "30d": k30}
        print(f"\n--- {r['coin']}  {fmt_price(r['price'])}  HYPE {r['hype_score']}/100  [{r['stage']}] ---")
        print(f"Why: {r['points']}")
        print(OPINION[r["stage"]])
        print(f"Risk (simulated from its own past swings): 7d down >10%: {k7['chance_down_10']}% | "
              f"30d down >25%: {k30['chance_down_25']}% | 30d up >25%: {k30['chance_up_25']}% | "
              f"bad case 30d: {k30['worst_5pct']:+}%")

    print("\nHype is not a strategy. Most coins that trend on social media are lower a month later. "
          "Not financial advice.")
    if args.demo:
        print("!! DEMO MODE - fake data.")

    if (args.notify or args.notify_all) and top:
        top5 = [r["coin"] for r in top[:5]]
        if args.notify_all or top5 != history.get("last_top5"):
            body = "\n".join(
                f"{i}. {r['coin']} {r['hype_score']}/100 {r['stage']} | {r['mentions_24h']} mentions"
                f"{' | trending #' + str(r['trending_rank']) if r['trending_rank'] else ''}"
                f" | 24h {r['change_24h']:+}%" for i, r in enumerate(top[:5], 1))
            early = [r["coin"] for r in top[:5] if r["stage"] == "EARLY"]
            if early:
                body += f"\n\nEarly-stage: {', '.join(early)}. Late = already pumped, be careful."
            core.notify("Hype radar: " + ", ".join(top5), body, tags="fire")
        history["last_top5"] = top5

    if not args.demo:
        history["runs"].append({"time": int(time.time()), "mentions": mentions,
                                "top": [{k: r[k] for k in ("coin", "hype_score", "stage", "price")}
                                        for r in top[:10]]})
        history["runs"] = [r for r in history["runs"] if r["time"] > time.time() - 30 * DAY]
        HISTORY_FILE.write_text(json.dumps(history))


if __name__ == "__main__":
    main()
