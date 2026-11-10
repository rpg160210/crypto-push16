"""Ask the analyst about coins.

  python analyze.py bitcoin solana pepe     # analyze specific coins (CoinGecko IDs)
  python analyze.py                         # analyze your watchlist (COINS env var)
  python analyze.py --notify                # also push results to your phone
  python analyze.py bitcoin --demo          # fake data, no internet

For each coin you get:
  * a verdict (STRONG BUY / BUY / HOLD / SELL / STRONG SELL) and an opinion
  * the trained model's chance the price is higher in 7 days
  * how each of the 8 strategies votes, weighted by how well it did in back-tests
  * a risk table from 20,000 simulated futures for the next 7 and 30 days
"""

import argparse
import json
import os
import sys
import time
import zlib

import joblib
import numpy as np
import pandas as pd

import core

STATE_FILE = core.ROOT / "state.json"
MAX_DETAILED = 10         # full reports for this many coins when watching a big list
MAX_DETAILED_PUSHES = 3   # detailed phone alerts per run on a big list (plus 1 summary)
VERDICTS = ["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"]


def load_bundle():
    path = core.ROOT / "model.joblib"
    if not path.exists():
        sys.exit("No trained model yet - run: python train.py  (or python train.py --demo)")
    return joblib.load(path)


def analyze_coin(coin, bundle, btc, load):
    df = load(coin)
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    if len(df) < 120:
        raise ValueError("listed less than ~4 months ago, not enough history")
    feat = core.build_features(df, btc).replace([np.inf, -np.inf], np.nan)
    today = feat.iloc[[-1]]
    p_up = float(bundle["model"].predict_proba(today[bundle["columns"]])[:, 1][0])

    votes, consensus = {}, 0.0
    for name, w in bundle["strategy_weights"].items():
        sig = int(today[f"sig_{name}"].iloc[0])
        votes[name] = {"vote": {1: "BUY", -1: "SELL", 0: "no signal"}[sig], "weight": w}
        consensus += w * sig

    # Model counts 60%, strategy vote 40%. Both range -1..+1.
    trusted = bundle.get("model_trusted", True)
    if trusted:
        score = 0.6 * (p_up - 0.5) * 2 / 0.3 + 0.4 * consensus  # /0.3: 65% chance up ~= full marks
    else:
        score = 0.7 * consensus  # model failed its exam -> strategies only (STRONG needs near-full agreement)
    score = float(np.clip(score, -1, 1))
    idx = 0 if score <= -0.5 else 1 if score <= -0.15 else 2 if score < 0.15 else 3 if score < 0.5 else 4

    tilt = p_up if trusted else None  # untrusted model -> neutral simulations
    risk7, risk30 = (core.risk_table(df["close"], 7, tilt),
                     core.risk_table(df["close"], 30, tilt))
    d = risk30["chance_down_25"]
    risk_level = "LOW" if d < 5 else "MEDIUM" if d < 15 else "HIGH" if d < 30 else "EXTREME"

    f = today.iloc[0]
    return {
        "coin": coin,
        "price": float(df["close"].iloc[-1]),
        "verdict": VERDICTS[idx],
        "score": round(score, 2),
        "model_trusted": trusted,
        "model_chance_up_7d": round(p_up * 100, 1) if trusted else None,
        "strategy_consensus": round(consensus, 2),
        "votes": votes,
        "risk_level": risk_level,
        "risk_7d": risk7,
        "risk_30d": risk30,
        "facts": {"rsi": round(float(f["rsi"]), 1), "change_7d": round(float(f["ret_7d"]) * 100, 1),
                  "change_30d": round(float(f["ret_30d"]) * 100, 1),
                  "daily_volatility": round(float(f["vol_30d"]) * 100, 1),
                  "below_90d_high": round(float(f["drawdown_90d"]) * 100, 1)},
    }


def rule_opinion(r):
    """A blunt opinion written from the numbers (used when Claude isn't set up)."""
    buys = [k for k, v in r["votes"].items() if v["vote"] == "BUY" and v["weight"] > 0]
    sells = [k for k, v in r["votes"].items() if v["vote"] == "SELL" and v["weight"] > 0]
    p, fx, rk = r["model_chance_up_7d"], r["facts"], r["risk_30d"]
    lead = {
        "STRONG BUY": "I like this one a lot.",
        "BUY": "I'm leaning bullish here.",
        "HOLD": "I'd stay out or sit tight - the signals are fighting each other.",
        "SELL": "I don't like how this looks.",
        "STRONG SELL": "I'd get out of this one.",
    }[r["verdict"]]
    n = lambda xs: f"{len(xs)} {'trusted strategy says' if len(xs) == 1 else 'trusted strategies say'}"
    why = ((f" The model gives it a {p}% chance of being higher in a week; " if p is not None
            else " Going by the strategies that have worked lately, ")
           + f"{n(buys)} buy ({', '.join(buys) or 'none'}) and "
           f"{n(sells)} sell ({', '.join(sells) or 'none'}).")
    if fx["rsi"] > 75:
        why += f" RSI is {fx['rsi']}, it's overheated - chasing it now is how people get burned."
    elif fx["rsi"] < 25:
        why += f" RSI is {fx['rsi']}, very oversold, so a bounce is possible but falling knives cut."
    warn = (f" Risk is {r['risk_level']}: in simulations it fell 25%+ within 30 days "
            f"{rk['chance_down_25']}% of the time.")
    return lead + why + warn


def claude_opinions(results):
    """Have Claude write an opinionated take on each coin, grounded in the numbers."""
    import anthropic

    schema = {"type": "object", "additionalProperties": False, "required": ["takes"],
              "properties": {"takes": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["coin", "opinion", "biggest_risk"],
                  "properties": {"coin": {"type": "string"}, "opinion": {"type": "string"},
                                 "biggest_risk": {"type": "string"}}}}}}
    slim = [{k: v for k, v in r.items()} for r in results]
    resp = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=(
            "You are a seasoned, blunt crypto trader writing quick takes for a friend. Each coin "
            "comes with a verdict from a trained model + back-tested strategies, and Monte Carlo "
            "risk percentages. For every coin write 2-3 sentences with a real opinion - say what "
            "you'd actually do and why, citing the most important numbers. Don't contradict the "
            "verdict without saying why. Then name the single biggest risk in one sentence. "
            "Plain language, no hype, no disclaimers (the app shows those)."),
        messages=[{"role": "user", "content": json.dumps(slim)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined")
    text = next(b.text for b in resp.content if b.type == "text")
    return {t["coin"]: t for t in json.loads(text)["takes"]}


def up(r):
    return f"{r['model_chance_up_7d']}%" if r["model_trusted"] else "n/a"


def fmt_price(p):
    return f"${p:,.2f}" if p >= 1 else f"${p:.6f}"


def print_table(results):
    print(f"\n{'coin':12}{'price':>14}  {'verdict':12}{'7d up':>7}  {'risk':8}"
          f"{'30d -25%':>9}{'30d +25%':>9}")
    for r in results:
        k = r["risk_30d"]
        print(f"{r['coin'].upper():12}{fmt_price(r['price']):>14}  {r['verdict']:12}"
              f"{up(r):>7}  {r['risk_level']:8}"
              f"{k['chance_down_25']:>8}%{k['chance_up_25']:>8}%")


def print_report(r):
    k7, k30 = r["risk_7d"], r["risk_30d"]
    bar = "=" * 64
    print(f"\n{bar}\n{r['coin'].upper()}  {fmt_price(r['price'])}    "
          f"VERDICT: {r['verdict']}    RISK: {r['risk_level']}\n{bar}")
    print(f"Opinion: {r['opinion']}")
    if r.get("biggest_risk"):
        print(f"Biggest risk: {r['biggest_risk']}")
    if r["model_trusted"]:
        print(f"\nModel: {r['model_chance_up_7d']}% chance it's higher in 7 days")
    else:
        print("\nModel: switched off (failed its last exam) - verdict uses strategies only")
    print("Strategy votes:", ", ".join(f"{k}={v['vote']}" for k, v in r["votes"].items()))
    print(f"\nWhat could happen - {k7['simulations']:,} simulated futures using this coin's own")
    print("past swings, tilted toward the model's 7-day view:" if r["model_trusted"]
          else "past swings (no direction assumed, since the model is switched off):")
    rows = [("Ends higher", "chance_up"), ("Ends lower", "chance_down"),
            ("Up more than 10%", "chance_up_10"), ("Up more than 25%", "chance_up_25"),
            ("Down more than 10%", "chance_down_10"), ("Down more than 25%", "chance_down_25"),
            ("Down more than 50%", "chance_down_50"),
            ("Drops 20% at some point", "chance_drop_20_at_some_point")]
    print(f"  {'':26}{'next 7 days':>13}{'next 30 days':>14}")
    for label, key in rows:
        print(f"  {label:26}{k7[key]:>12}%{k30[key]:>13}%")
    print(f"  {'Typical range (middle 50%)':26}{k7['typical_range'][0]:>+6}..{k7['typical_range'][1]:+}%"
          f"{k30['typical_range'][0]:>+7}..{k30['typical_range'][1]:+}%")
    print(f"  {'Bad case (worst 5%)':26}{k7['worst_5pct']:>+12}%{k30['worst_5pct']:>+13}%")
    print(f"  {'Good case (best 5%)':26}{k7['best_5pct']:>+12}%{k30['best_5pct']:>+13}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coins", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--notify", action="store_true", help="push verdict changes to your phone")
    ap.add_argument("--notify-all", action="store_true", help="push every coin, changed or not")
    args = ap.parse_args()

    wanted = args.coins or [c.strip() for c in os.getenv(
        "COINS", "bitcoin,ethereum,solana,ripple,dogecoin").split(",") if c.strip()]
    bundle = load_bundle()

    binance_mode = [w.lower() for w in wanted] == ["binance"]
    if args.demo:
        coins = ([f"DEMO{i}" for i in range(60)] if binance_mode else wanted)
        load = lambda c: core.fake_history(zlib.crc32(c.encode()))
        btc = core.fake_history(0)
    elif binance_mode:
        coins = core.binance_coins()
        print(f"Analyzing all {len(coins)} coins trading on Binance (USDT pairs)...")
        load = core.binance_history
        btc = core.binance_history("BTC")
    else:
        coins = wanted
        btc = core.history("bitcoin", max_age_h=3)

        def load(c):
            df = core.history(c, max_age_h=3)
            time.sleep(2.2 if core.CG_KEY else 6)  # CoinGecko free rate limit
            return df

    results, skipped = [], []
    for coin in coins:
        try:
            results.append(analyze_coin(coin, bundle, btc, load))
        except Exception as e:
            skipped.append(coin)
            hint = "" if binance_mode else " (is that the CoinGecko ID? e.g. 'ripple' not 'xrp')"
            print(f"!! skipped {coin}: {e}{hint}", file=sys.stderr)
    results.sort(key=lambda r: -r["score"])

    # Big lists: full write-ups only for the strongest calls, a table for everything.
    big = len(results) > 20
    strong = [r for r in results if "STRONG" in r["verdict"]]
    detailed = (sorted(strong, key=lambda r: -abs(r["score"]))[:MAX_DETAILED]) if big else results

    takes = {}
    if os.getenv("ANTHROPIC_API_KEY") and detailed:
        try:
            takes = claude_opinions(detailed)
        except Exception as e:
            print(f"!! Claude unavailable, using built-in opinions: {e}", file=sys.stderr)
    for r in results:
        t = takes.get(r["coin"], {})
        r["opinion"] = t.get("opinion") or rule_opinion(r)
        r["biggest_risk"] = t.get("biggest_risk", "")

    if big:
        print_table(results)
        print(f"\nDetailed reports for the {len(detailed)} strongest calls:")
    for r in detailed:
        print_report(r)
    if skipped:
        print(f"\nSkipped {len(skipped)} coins: {', '.join(skipped)}")

    if bundle.get("demo") or args.demo:
        print("\n!! DEMO MODE - fake data. Do not use for real decisions.")
    print("\nNot financial advice. The model is right more often than a coin flip in back-tests, "
          "but it is wrong a lot, and crypto can drop more than any simulation shows.")

    if args.notify or args.notify_all:
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        first_run = not state
        if big:
            notify_big(results, state, args.notify_all, first_run)
        else:
            for r in results:
                if args.notify_all or state.get(r["coin"]) != r["verdict"]:
                    push_detail(r)
        state = {r["coin"]: r["verdict"] for r in results} | {
            c: v for c, v in state.items() if c in skipped}
        if not args.demo:
            STATE_FILE.write_text(json.dumps(state, indent=2))


def push_detail(r):
    k = r["risk_30d"]
    tag = {"STRONG BUY": "rocket", "BUY": "chart_with_upwards_trend", "HOLD": "pause_button",
           "SELL": "chart_with_downwards_trend", "STRONG SELL": "rotating_light"}[r["verdict"]]
    core.notify(
        f"{r['verdict']}: {r['coin'].upper()} @ {fmt_price(r['price'])} | risk {r['risk_level']}",
        f"{r['opinion']}\n\n7d: {up(r)} chance up\n"
        f"30d: {k['chance_up_10']}% chance +10% | {k['chance_down_10']}% chance -10% | "
        f"{k['chance_down_25']}% chance -25%",
        priority="high" if "STRONG" in r["verdict"] else "default", tags=tag)


def notify_big(results, state, notify_all, first_run):
    """For big lists: one summary push + detailed pushes for a few of the strongest
    NEW calls, instead of hundreds of notifications."""
    if notify_all or first_run:
        new = [r for r in results if "STRONG" in r["verdict"]]
    else:
        new = [r for r in results if "STRONG" in r["verdict"] and state.get(r["coin"]) != r["verdict"]]
    if not new:
        return
    line = lambda r: (f"{r['coin'].upper()} {r['verdict'].split()[-1].lower()} score {r['score']:+}, "
                      f"risk {r['risk_level']}")
    buys = [r for r in new if r["verdict"] == "STRONG BUY"]
    sells = [r for r in new if r["verdict"] == "STRONG SELL"]
    body = ""
    if buys:
        body += f"STRONG BUY ({len(buys)}):\n" + "\n".join(line(r) for r in buys[:15]) + "\n\n"
    if sells:
        body += (f"STRONG SELL ({len(sells)}):\n"
                 + "\n".join(line(r) for r in sorted(sells, key=lambda r: r["score"])[:15]))
    core.notify(f"{'Top' if first_run else 'New'} strong signals: {len(buys)} buy, {len(sells)} sell "
                f"(of {len(results)} coins)", body.strip(), priority="high", tags="bell")
    for r in sorted(new, key=lambda r: -abs(r["score"]))[:MAX_DETAILED_PUSHES]:
        push_detail(r)


if __name__ == "__main__":
    main()
