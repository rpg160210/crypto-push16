"""Builds docs/index.html - one page with everything the bot knows - and sends the
weekly summary push. Runs automatically after every bot workflow (dashboard.yml).

  python dashboard.py            # build the page (+ weekly push when it's due)
  python dashboard.py --weekly   # force the weekly push now
"""

import argparse
import html
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import core

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
SUMMARY_STATE = DOCS / "summary_state.json"
DAY_MS = 86_400_000


def load(name, default=None):
    try:
        return json.loads((ROOT / name).read_text())
    except Exception:
        return default


def ts(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%b %d %H:%M")


def esc(x):
    return html.escape(str(x))


def live_prices():
    try:
        return {t["symbol"][:-4]: float(t["price"]) for t in core._binance_get("ticker/price")
                if t["symbol"].endswith("USDT")}
    except Exception:
        return {}


# ------------------------------------------------------------------ numbers ---
def trade_stats(trades, practice, days=30, now=None):
    now = now or time.time() * 1000
    done = [t for t in trades if t["status"] != "open" and bool(t.get("practice")) == practice
            and t["opened"] >= now - days * DAY_MS]
    if not done:
        return None
    nets = [t["net"] for t in done]
    return {"n": len(done), "win": 100 * sum(n > 0 for n in nets) / len(nets),
            "avg": 100 * sum(nets) / len(nets), "sum": 100 * sum(nets)}


def curve(trades, practice):
    """Cumulative % result of closed trades, in the order they closed."""
    done = sorted((t for t in trades if t["status"] != "open" and bool(t.get("practice")) == practice),
                  key=lambda t: t.get("closed", t["opened"]))
    pts, total = [], 0.0
    for t in done:
        total += 100 * t["net"]
        pts.append((t.get("closed", t["opened"]), total))
    return pts


def btc_curve(trades, now_price):
    """What just holding Bitcoin did over the same time (from BTC prices saved with each trade)."""
    pts = sorted((t["opened"], t["btc_entry"]) for t in trades if t.get("btc_entry"))
    if not pts:
        return []
    base = pts[0][1]
    out = [(t, 100 * (p / base - 1)) for t, p in pts]
    if now_price:
        out.append((time.time() * 1000, 100 * (now_price / base - 1)))
    return out


# ------------------------------------------------------------------- charts ---
def line_chart(series, title, ref=(), height=220, pct=True):
    """Small dependency-free SVG line chart. series = [(name, css_class, [(x, y), ...])]."""
    pts = [p for _, _, s in series for p in s]
    if len({p[0] for p in pts}) < 2:
        return f'<div class="empty">{esc(title)}: not enough data yet - fills in as trades finish.</div>'
    W, H, L, R, T, B = 480, height, 58, 10, 14, 28
    xs, ys = [p[0] for p in pts], [p[1] for p in pts] + [r for r, _ in ref]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad
    X = lambda x: L + (x - x0) / ((x1 - x0) or 1) * (W - L - R)
    Y = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for i in range(5):  # gridlines + labels
        v = y0 + (y1 - y0) * i / 4
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="axis" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">'
                 f'{v:+.1f}{"%" if pct else ""}</text>' if pct else
                 f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="axis" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.3f}</text>')
    for v, label in ref:
        g.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="reflabel" x="{W - R - 4}" y="{Y(v) - 4:.1f}" text-anchor="end">{esc(label)}</text>')
    for x, anchor in ((x0, "start"), (x1, "end")):
        g.append(f'<text class="axis" x="{X(x):.1f}" y="{H - 6}" text-anchor="{anchor}">{ts(x)}</text>')
    for name, cls, s in series:
        s = sorted(s)
        if len(s) == 1:
            g.append(f'<circle class="{cls}" cx="{X(s[0][0]):.1f}" cy="{Y(s[0][1]):.1f}" r="3"/>')
        elif s:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(s))
            g.append(f'<path class="line {cls}" d="{d}"/>')
    g.append("</svg>")
    legend = "".join(f'<span class="key {cls}">{esc(n)}</span>' for n, cls, s in series if s)
    return f'<figure><figcaption>{esc(title)}</figcaption>{"".join(g)}<div class="legend">{legend}</div></figure>'


def date_ms(s):
    try:
        return datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000
    except Exception:
        try:
            return datetime.fromisoformat(s).timestamp() * 1000
        except Exception:
            return None


# --------------------------------------------------------------------- page ---
CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--good:#1f8a4c;--bad:#c2412d;
--warn:#b7791f;--a:#2f6fdb;--b:#9a5bd4;--c:#d08a1e;--grid:#ecebe6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#a3a29b;
--line:#33332f;--good:#4cc27e;--bad:#ef7a64;--warn:#e0b15a;--a:#7aa7ff;--b:#c497f2;--c:#f0b356;--grid:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 2px}
h2{font-size:16px;margin:28px 0 10px}.sub{color:var(--muted);font-size:13px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--muted);font-size:12px}.card .v{font-size:20px;font-weight:600;margin-top:2px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;
overflow:hidden;font-size:13.5px}th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500;font-size:12px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}figure{margin:0 0 12px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 12px}figcaption{font-size:13px;color:var(--muted);margin-bottom:4px}
svg{width:100%;height:auto;display:block}.grid{stroke:var(--grid)}.axis{fill:var(--muted);font-size:13px}
.ref{stroke:var(--muted);stroke-dasharray:4 4}.reflabel{fill:var(--muted);font-size:13px}
path.line{fill:none;stroke-width:2.5}.a{stroke:var(--a);fill:var(--a);color:var(--a)}
.b{stroke:var(--b);fill:var(--b);color:var(--b)}.c{stroke:var(--c);fill:var(--c);color:var(--c)}
.legend{display:flex;gap:14px;font-size:12px;margin-top:4px}.key::before{content:"";display:inline-block;
width:10px;height:3px;background:currentColor;margin-right:6px;vertical-align:middle}
.empty{color:var(--muted);font-size:13px;padding:12px;background:var(--card);border:1px dashed var(--line);
border-radius:10px;margin-bottom:12px}.pill{display:inline-block;padding:1px 8px;border-radius:99px;
font-size:12px;border:1px solid var(--line)}.foot{color:var(--muted);font-size:12px;margin-top:32px}
"""


def card(k, v, cls=""):
    return f'<div class="card"><div class="k">{esc(k)}</div><div class="v {cls}">{v}</div></div>'


def signed(x, digits=2, unit="%"):
    if x is None:
        return "-"
    cls = "good" if x > 0 else "bad" if x < 0 else ""
    return f'<span class="{cls}">{x:+.{digits}f}{unit}</span>'


def build_page():
    tr = load("training_report.json", {}) or {}
    dr = load("daytrade_report.json", {}) or {}
    j = load("trade_journal.json", {"trades": []}) or {"trades": []}
    state = load("state.json", {}) or {}
    hype = load("hype_history.json", {}) or {}
    trades = j.get("trades", [])
    prices = live_prices()
    now = time.time() * 1000

    daily_on = (tr.get("exam") or {}).get("model_trusted")
    dt_edge = (dr.get("exam") or {}).get("signals")
    safety = (j.get("safety") or {}).get("paused")
    real, prac = trade_stats(trades, False, now=now), trade_stats(trades, True, now=now)
    review = tr.get("call_review") or {}

    top = "".join([
        card("Daily model", ("ON" if daily_on else "OFF - strategies only"), "good" if daily_on else "warn"),
        card("Day-trade signals", ("ON - edge found" if dt_edge else "OFF - no proven edge"),
             "good" if dt_edge else "warn"),
        card("Safety switch", "PAUSED" if safety else "OK", "bad" if safety else "good"),
        card("Past calls right (7d later)",
             f"{review['overall_right_pct']}% of {review['reviewed']}" if review.get("reviewed") else "waiting"),
    ])

    # --- day trading ---
    dt_cards = "".join([
        card("Real signals, 30 days", f"{real['n']} trades · {real['win']:.0f}% won · {signed(real['avg'])} avg"
             if real else "none yet"),
        card("Practice trades, 30 days", f"{prac['n']} trades · {prac['win']:.0f}% won · {signed(prac['avg'])} avg"
             if prac else "none yet"),
    ])
    chart_dt = line_chart(
        [("Practice trades (sum of results)", "a", curve(trades, True)),
         ("Real signals (sum of results)", "b", curve(trades, False)),
         ("Just holding Bitcoin", "c", btc_curve(trades, prices.get("BTC")))],
        "Day trading: cumulative result after costs vs. holding Bitcoin", ref=((0, "0"),))

    open_rows = []
    for t in sorted((t for t in trades if t["status"] == "open"), key=lambda t: -t["opened"]):
        p = prices.get(t["coin"])
        live = 100 * (p / t["entry"] - 1) - 100 * core_cost() if p else None
        open_rows.append(
            f"<tr><td>{esc(t['coin'])} <span class='pill'>{'practice' if t.get('practice') else 'signal'}</span></td>"
            f"<td class='n'>{t['entry']:.6g}</td><td class='n'>{t['tp']:.6g}</td><td class='n'>{t['sl']:.6g}</td>"
            f"<td class='n'>{signed(live)}</td><td>{ts(t['opened'])}</td><td>{ts(t['deadline'])}</td></tr>")
    open_tbl = ("<div class='scroll'><table><tr><th>Coin</th><th class='n'>Entry</th><th class='n'>Take-profit</th>"
                "<th class='n'>Stop-loss</th><th class='n'>Now (after costs)</th><th>Opened (UTC)</th>"
                "<th>Closes by</th></tr>" + "".join(open_rows) + "</table></div>") if open_rows else \
        '<div class="empty">No open trades right now.</div>'

    closed = sorted((t for t in trades if t["status"] != "open"), key=lambda t: -t.get("closed", t["opened"]))[:15]
    closed_tbl = ("<div class='scroll'><table><tr><th>Coin</th><th>Result</th><th class='n'>After costs</th>"
                  "<th>Opened (UTC)</th></tr>" + "".join(
                      f"<tr><td>{esc(t['coin'])} <span class='pill'>{'practice' if t.get('practice') else 'signal'}"
                      f"</span></td><td>{esc(t['status'])}</td><td class='n'>{signed(100 * t['net'])}</td>"
                      f"<td>{ts(t['opened'])}</td></tr>" for t in closed) + "</table></div>") if closed else ""

    # --- past calls review ---
    rows = []
    for k, v in (review.get("by_call") or {}).items():
        kind, call = k.split(":", 1)
        beat = f"{v['beat_btc']}/{v['beat_btc_of']}" if "beat_btc" in v else "-"
        rows.append(f"<tr><td>{esc(kind)}</td><td>{esc(call)}</td><td class='n'>{v['calls']}</td>"
                    f"<td class='n'>{v['right_pct']}%</td><td class='n'>{signed(v['avg_7d_return_pct'])}</td>"
                    f"<td class='n'>{signed(v.get('btc_same_weeks_pct'))}</td><td class='n'>{beat}</td></tr>")
    calls_tbl = ("<div class='scroll'><table><tr><th>Mode</th><th>Call</th><th class='n'>Calls</th>"
                 "<th class='n'>Right</th><th class='n'>Avg 7d after</th><th class='n'>BTC same weeks</th>"
                 "<th class='n'>Beat BTC</th></tr>" + "".join(rows) + "</table></div>") if rows else \
        '<div class="empty">No calls are 7 days old yet.</div>'

    # --- model scores ---
    chart_auc = line_chart(
        [("Daily model", "a", [(date_ms(h["date"]), h["auc"]) for h in tr.get("history", []) if date_ms(h["date"])]),
         ("Day-trade model", "b", [(date_ms(h["date"]), h["auc"]) for h in dr.get("history", []) if date_ms(h["date"])])],
        "Exam score after each training (AUC on unseen recent data)",
        ref=((0.5, "coin flip"), (0.53, "switch-on level")), pct=False)

    # --- today's calls ---
    strong_buy = sorted(c for c, v in state.items() if v == "STRONG BUY")
    counts = {v: sum(1 for x in state.values() if x == v)
              for v in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")}
    buys = sorted(c for c, v in state.items() if v in ("STRONG BUY", "BUY"))
    last_hype = (hype.get("runs") or [{}])[-1].get("top", []) if hype.get("runs") else []
    hype_rows = "".join(f"<tr><td>{esc(h['coin'])}</td><td class='n'>{h['hype_score']}</td>"
                        f"<td>{esc(h['stage'])}</td></tr>" for h in last_hype[:10])
    today = ("<div class='grid4'>" + "".join(card(k, n, "good" if "BUY" in k else "bad" if "SELL" in k else "")
                                              for k, n in counts.items()) + "</div>"
             f"<p class='sub'>Buy-rated now: {esc(', '.join(buys[:60]) or 'none')}</p>"
             + (f"<h2>Hype radar (latest)</h2><div class='scroll'><table><tr><th>Coin</th><th class='n'>Hype</th>"
                f"<th>Stage</th></tr>{hype_rows}</table></div>" if hype_rows else ""))

    weather_txt = ""
    for label, w in (("Day trading", (dr.get("exam") or {}).get("market_filter")),
                     ("Daily calls", tr.get("market_filter"))):
        if w:
            weather_txt += (f"<p class='sub'>Market-weather filter ({label}): "
                            f"<b>{'ON' if w.get('enabled') else 'OFF'}</b> - {esc(w.get('why', ''))}</p>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Crypto Bot Dashboard</title>
<style>{CSS}</style></head><body><main>
<h1>Crypto bot dashboard</h1>
<div class="sub">Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC · paper trading only ·
not financial advice</div>
<div class="grid4">{top}</div>
<h2>Day trading</h2><div class="grid4">{dt_cards}</div>{weather_txt}
{chart_dt}
<h2>Open trades</h2>{open_tbl}
{"<h2>Recently closed</h2>" + closed_tbl if closed_tbl else ""}
<h2>Past calls, checked 7 days later</h2>{calls_tbl}
<p class="sub">Right = buy calls went up / sell calls went down. Beat BTC is the stricter test:
buy calls rose more than Bitcoin, sell calls fell more than Bitcoin.</p>
<h2>Model scores</h2>{chart_auc}
<h2>Today's calls</h2>{today}
<p class="foot">Built from the bot's files in this repository. Every number is paper trading - nothing is bought.</p>
</main></body></html>"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page)
    (DOCS / ".nojekyll").write_text("")
    print(f"Dashboard written: docs/index.html ({len(page) // 1024} KB)")
    return {"real": real, "prac": prac, "review": review, "daily_on": daily_on, "dt_edge": dt_edge,
            "safety": safety, "trades": trades, "prices": prices, "strong_buy": strong_buy}


def core_cost():
    try:
        import daytrade
        return daytrade.COST
    except Exception:
        return 0.0025


# ------------------------------------------------------------- weekly push ---
def weekly_due(force):
    st = json.loads(SUMMARY_STATE.read_text()) if SUMMARY_STATE.exists() else {}
    now = datetime.now(timezone.utc)
    last = st.get("last_sent", 0)
    sunday_evening = now.weekday() == 6 and now.hour >= 17
    return force or (sunday_evening and time.time() - last > 3 * 86400) or (time.time() - last > 8 * 86400 and last)


def send_weekly(info):
    t, now = info["trades"], time.time() * 1000
    r7, p7 = trade_stats(t, False, 7, now), trade_stats(t, True, 7, now)
    btc = btc_curve([x for x in t if x["opened"] >= now - 7 * DAY_MS], info["prices"].get("BTC"))
    lines = [f"Daily model: {'ON' if info['daily_on'] else 'OFF (strategies only)'} | "
             f"Day-trade signals: {'ON' if info['dt_edge'] else 'OFF (no edge)'}"
             + (" | SAFETY SWITCH ON" if info["safety"] else "")]
    if r7:
        lines.append(f"Day-trade signals this week: {r7['n']} trades, {r7['win']:.0f}% won, {r7['sum']:+.1f}% total")
    if p7:
        lines.append(f"Practice trades this week: {p7['n']}, {p7['win']:.0f}% won, {p7['avg']:+.2f}% avg")
    if btc:
        lines.append(f"Bitcoin this week: {btc[-1][1]:+.1f}%")
    rv = info["review"]
    if rv.get("reviewed"):
        lines.append(f"Past calls: {rv['overall_right_pct']}% of {rv['reviewed']} right")
        best = sorted(rv["by_call"].items(), key=lambda kv: -kv[1]["calls"])[:4]
        lines += [f"  {k.split(':', 1)[1]}: {v['right']}/{v['calls']} right"
                  + (f", beat BTC {v['beat_btc']}/{v['beat_btc_of']}" if "beat_btc" in v else "") for k, v in best]
    url = os.getenv("DASHBOARD_URL", "")
    if url:
        lines.append(f"\nDashboard: {url}")
    core.notify("Weekly crypto bot summary", "\n".join(lines), tags="calendar")
    DOCS.mkdir(exist_ok=True)
    SUMMARY_STATE.write_text(json.dumps({"last_sent": time.time()}))
    print("Weekly summary sent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true", help="send the weekly summary now")
    args = ap.parse_args()
    info = build_page()
    if weekly_due(args.weekly):
        send_weekly(info)


if __name__ == "__main__":
    main()
