"""Builds docs/index.html - the bot's dashboard - and sends the weekly summary push.
Runs automatically after every bot workflow (dashboard.yml).

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


def trade_cost():
    try:
        import daytrade
        return daytrade.COST
    except Exception:
        return 0.0025


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
    done = sorted((t for t in trades if t["status"] != "open" and bool(t.get("practice")) == practice),
                  key=lambda t: t.get("closed", t["opened"]))
    pts, total = [], 0.0
    for t in done:
        total += 100 * t["net"]
        pts.append((t.get("closed", t["opened"]), total))
    return pts


def btc_curve(trades, now_price):
    pts = sorted((t["opened"], t["btc_entry"]) for t in trades if t.get("btc_entry"))
    if not pts:
        return []
    base = pts[0][1]
    out = [(t, 100 * (p / base - 1)) for t, p in pts]
    if now_price:
        out.append((time.time() * 1000, 100 * (now_price / base - 1)))
    return out


def date_ms(s):
    for fmt in ("%Y-%m-%d %H:%M",):
        try:
            return datetime.strptime(s[:16], fmt).replace(tzinfo=timezone.utc).timestamp() * 1000
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s).timestamp() * 1000
    except Exception:
        return None


# ------------------------------------------------------------------- charts ---
def line_chart(series, title, ref=(), height=230, pct=True, note=""):
    """Dependency-free SVG line chart. series = [(name, css_class, [(x, y), ...])]."""
    pts = [p for _, _, s in series for p in s]
    if len({p[0] for p in pts}) < 2:
        return (f'<div class="empty"><b>{esc(title)}</b><br>Not enough data yet - this fills in '
                f'automatically as the bot runs.</div>')
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
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        lab = f"{v:+.1f}%" if pct else f"{v:.3f}"
        g.append(f'<line class="gl" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="ax" x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    for v, label in ref:
        g.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="ax" x="{W - R - 4}" y="{Y(v) - 5:.1f}" text-anchor="end">{esc(label)}</text>')
    for x, anchor in ((x0, "start"), (x1, "end")):
        g.append(f'<text class="ax" x="{X(x):.1f}" y="{H - 6}" text-anchor="{anchor}">{ts(x)}</text>')
    for name, cls, s in series:
        s = sorted(s)
        if len(s) == 1:
            g.append(f'<circle class="{cls}" cx="{X(s[0][0]):.1f}" cy="{Y(s[0][1]):.1f}" r="3.5"/>')
        elif s:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(s))
            g.append(f'<path class="ln {cls}" d="{d}"/>')
    g.append("</svg>")
    legend = "".join(f'<span class="key {cls}">{esc(n)}</span>' for n, cls, s in series if s)
    return (f'<figure class="card"><figcaption>{esc(title)}</figcaption>{"".join(g)}'
            f'<div class="legend">{legend}</div>{f"<p class=hint>{note}</p>" if note else ""}</figure>')


# --------------------------------------------------------------- html bits ---
def stat(label, value, sub="", cls=""):
    return (f'<div class="stat"><div class="lbl">{label}</div><div class="val {cls}">{value}</div>'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}</div>')


def signed(x, digits=2, unit="%"):
    if x is None:
        return '<span class="muted">-</span>'
    cls = "up" if x > 0 else "down" if x < 0 else ""
    return f'<span class="{cls}">{x:+.{digits}f}{unit}</span>'


def badge(text, kind):
    return f'<span class="badge {kind}">{esc(text)}</span>'


def section(sid, title, intro, body):
    return (f'<section id="{sid}" class="tab"><h2>{title}</h2>'
            f'{f"<p class=intro>{intro}</p>" if intro else ""}{body}</section>')


GUIDE = [
    ("Verdict", "The bot's call on a coin for the next ~7 days: STRONG BUY, BUY, HOLD, SELL or STRONG SELL. "
                "STRONG means the trusted strategies all agree."),
    ("Score", "From -1 (very bearish) to +1 (very bullish). It combines the strategies that worked best "
              "recently (and the AI model, when it has passed its exam)."),
    ("Chance up", "The AI model's estimate that the price is higher in 7 days. Shows '-' when the model is "
                  "switched off because it couldn't beat a coin flip on its last exam."),
    ("Risk level", "From 20,000 simulated futures based on the coin's own past swings: LOW, MEDIUM, HIGH or "
                   "EXTREME, by the chance of falling 25%+ within 30 days."),
    ("Strategies", "Classic trading rules (trend following, momentum, MACD, Turtle breakout, ...). Only the "
                   "ones that made money in recent months get a vote."),
    ("Practice trade / call", "The bot's top picks that weren't strong enough to alert you. Tracked like real "
                              "ones so it can learn from them. No alerts are sent for these."),
    ("Paper trading", "Every trade here is pretend - nothing is bought. It's how the bot proves itself "
                      "before you risk real money."),
    ("After costs", "Day-trade results subtract 0.25% per trade for Binance fees and slippage."),
    ("Beat BTC", "The honest test. A buy call beats Bitcoin if the coin rose more than Bitcoin did; a sell "
                 "call beats Bitcoin if the coin fell more. Being 'right' just because the whole market "
                 "moved doesn't count as skill."),
    ("Exam score (AUC)", "How well the model ranks winners above losers on recent data it never saw. 0.50 = "
                         "coin flip. Below 0.53 the model switches itself off."),
    ("Safety switch", "If the last ~25 real day-trade signals lost money on average, new signals pause "
                      "until a new training finds an edge again."),
    ("Market-weather filter", "If training finds that buys made while Bitcoin trends down lost money, buy "
                              "signals pause during Bitcoin downtrends."),
    ("Hype stage", "EARLY = buzz before the price moved. BUILDING = both rising. LATE = already pumped. "
                   "FADING = people talk about it less. MILD = no real hype."),
]

CSS = r"""
:root{--bg:#000;--panel:#0b0b0d;--card:#111114;--line:#232328;--ink:#ececf1;--muted:#8b8b96;
--blue:#3b82f6;--blue2:#93c5fd;--blue3:#1e3a8a;--up:#22c55e;--down:#ef4444;--warn:#f59e0b}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;-webkit-text-size-adjust:100%}
a{color:var(--blue2)}
header{position:sticky;top:0;z-index:10;background:rgba(0,0,0,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line)}
.bar{max-width:1080px;margin:0 auto;padding:10px 16px;display:flex;align-items:center;gap:12px}
.logo{font-weight:700;letter-spacing:.2px;white-space:nowrap}.logo b{color:var(--blue)}
.upd{color:var(--muted);font-size:12px;margin-left:auto;white-space:nowrap}
nav{max-width:1080px;margin:0 auto;padding:0 10px 8px;display:flex;gap:4px;overflow-x:auto;
scrollbar-width:none}nav::-webkit-scrollbar{display:none}
nav button{background:none;border:1px solid transparent;color:var(--muted);font:inherit;font-size:14px;
padding:6px 12px;border-radius:99px;cursor:pointer;white-space:nowrap}
nav button:hover{color:var(--ink)}nav button.on{color:#fff;background:var(--blue3);border-color:var(--blue)}
main{max-width:1080px;margin:0 auto;padding:18px 16px 60px}
.tab{display:none}.tab.on{display:block}
h2{font-size:20px;margin:6px 0 4px}h3{font-size:15px;margin:22px 0 8px;color:var(--blue2);font-weight:600}
.intro{color:var(--muted);margin:0 0 14px;max-width:760px}.hint{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.muted{color:var(--muted)}.up{color:var(--up)}.down{color:var(--down)}.warn{color:var(--warn)}.blue{color:var(--blue2)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:10px 0}
.stat,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat .lbl{color:var(--muted);font-size:12px}.stat .val{font-size:20px;font-weight:650;margin-top:2px}
.stat .sub{color:var(--muted);font-size:12px;margin-top:2px}.stat .val.sm{font-size:13.5px;font-weight:500;word-break:break-word}
.say{background:linear-gradient(135deg,#0b1730,#0b0b0d);border:1px solid var(--blue3);border-radius:12px;
padding:14px 16px;margin:6px 0 14px}.say p{margin:4px 0}.say b{color:var(--blue2)}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;
border:1px solid var(--line);white-space:nowrap}
.badge.sb{background:#052e16;border-color:#166534;color:#4ade80}.badge.b{color:#4ade80}
.badge.h{color:var(--muted)}.badge.s{color:#f87171}.badge.ss{background:#2f0a0a;border-color:#7f1d1d;color:#f87171}
.badge.pr{color:var(--blue2);border-color:var(--blue3)}.badge.on{background:var(--blue3);border-color:var(--blue);color:#fff}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
input[type=search],select{background:var(--card);border:1px solid var(--line);color:var(--ink);font:inherit;
font-size:14px;padding:8px 10px;border-radius:10px}input[type=search]{flex:1;min-width:160px}
input[type=search]:focus,select:focus{outline:none;border-color:var(--blue)}
.chip{background:var(--card);border:1px solid var(--line);color:var(--muted);font:inherit;font-size:13px;
padding:5px 11px;border-radius:99px;cursor:pointer}.chip.on{background:var(--blue3);border-color:var(--blue);color:#fff}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:12px;position:sticky;top:0;background:var(--card)}
th[data-k]{cursor:pointer}th[data-k]:hover{color:var(--blue2)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}
tr.row{cursor:pointer}tr.row:hover td{background:#16161b}
tr.det td{white-space:normal;background:#0d0d11;padding:12px 14px}
.dwrap{position:sticky;left:14px;max-width:calc(100vw - 62px)}.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.dgrid h4{margin:0 0 6px;font-size:12px;color:var(--blue2);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:2px 0}
.more{text-align:center;margin:10px 0}.more button{background:var(--card);border:1px solid var(--line);
color:var(--blue2);font:inherit;padding:7px 16px;border-radius:99px;cursor:pointer}
figure{margin:0 0 12px}figcaption{font-size:13px;color:var(--muted);margin-bottom:6px}
svg{width:100%;max-width:720px;height:auto;display:block;margin:0 auto}.gl{stroke:#1b1b20}.ax{fill:var(--muted);font-size:13px}
.ref{stroke:#4b5563;stroke-dasharray:4 4}path.ln{fill:none;stroke-width:2.5}
.c1{stroke:var(--blue);fill:var(--blue);color:var(--blue)}.c2{stroke:var(--blue2);fill:var(--blue2);color:var(--blue2)}
.c3{stroke:#9ca3af;fill:#9ca3af;color:#9ca3af}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;margin-top:6px}
.key::before{content:"";display:inline-block;width:12px;height:3px;background:currentColor;margin-right:6px;vertical-align:middle}
.empty{color:var(--muted);font-size:13.5px;padding:14px;background:var(--card);border:1px dashed var(--line);
border-radius:12px;margin-bottom:12px}
.bars .r{display:grid;grid-template-columns:130px 1fr 48px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.bars .t{height:8px;background:#1b1b20;border-radius:99px;overflow:hidden}.bars .f{height:100%;background:var(--blue)}
dl.g dt{font-weight:600;color:var(--blue2);margin-top:12px}dl.g dd{margin:2px 0 0;color:#c9c9d1}
footer{color:var(--muted);font-size:12px;text-align:center;padding:24px 16px}
@media(max-width:600px){.upd{display:none}.stat .val{font-size:18px}.bars .r{grid-template-columns:100px 1fr 44px}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s, el=document) => el.querySelector(s), $$ = (s, el=document) => [...el.querySelectorAll(s)];
// ---- menu ----
function show(id){ $$('.tab').forEach(t=>t.classList.toggle('on', t.id===id));
  $$('nav button').forEach(b=>b.classList.toggle('on', b.dataset.t===id)); window.scrollTo(0,0); }
$$('nav button').forEach(b=>b.onclick=()=>{ location.hash=b.dataset.t; });
$$('[data-go]').forEach(a=>a.onclick=e=>{ e.preventDefault(); location.hash=a.dataset.go; });
const ids = new Set($$('.tab').map(t=>t.id));
const fromHash = () => { const h=(location.hash||'#overview').slice(1); show(ids.has(h)?h:'overview'); };
window.addEventListener('hashchange', fromHash); fromHash();
// ---- "updated X min ago" ----
function ago(ms){ const m=Math.round((Date.now()-ms)/60000); return m<1?'just now':m<60?m+' min ago':m<1440?Math.round(m/60)+' h ago':Math.round(m/1440)+' d ago'; }
$$('[data-ago]').forEach(el=>el.textContent=ago(+el.dataset.ago));
// ---- calls table ----
const V = {'STRONG BUY':['sb',5],'BUY':['b',4],'HOLD':['h',3],'SELL':['s',2],'STRONG SELL':['ss',1]};
const RL = {LOW:1,MEDIUM:2,HIGH:3,EXTREME:4};
let f = {q:'', v:'ALL', r:'ALL', sort:'s', dir:-1, limit:60};
const fmtP = p => p>=1 ? '$'+p.toLocaleString(undefined,{maximumFractionDigits:2}) : '$'+p.toPrecision(4);
const pct = (x,d=1) => x==null ? '<span class="muted">-</span>' : `<span class="${x>0?'up':x<0?'down':''}">${x>0?'+':''}${x.toFixed(d)}%</span>`;
const nice = s => s.replace(/_/g,' ');
function rows(){
  let c = (D.analysis.coins||[]).filter(x => (f.v==='ALL' || x.v===f.v || (f.v==='BUYS' && x.v.includes('BUY')) || (f.v==='SELLS' && x.v.includes('SELL')))
    && (f.r==='ALL' || x.rl===f.r) && (!f.q || x.c.toLowerCase().includes(f.q)));
  const key = {s:x=>x.s, c:x=>x.c, p:x=>x.p, up:x=>x.up??-1, rl:x=>RL[x.rl], d25:x=>x.r30.chance_down_25, u25:x=>x.r30.chance_up_25, ch7:x=>x.f.change_7d}[f.sort];
  c.sort((a,b)=>{ const A=key(a),B=key(b); return (A>B?1:A<B?-1:0)*f.dir; });
  return c;
}
function draw(){
  const c = rows(), body = $('#calls-body');
  $('#calls-count').textContent = `${c.length} coin${c.length===1?'':'s'}`;
  body.innerHTML = c.slice(0,f.limit).map((x,i)=>`
   <tr class="row" data-i="${i}"><td><b>${x.c}</b></td><td><span class="badge ${V[x.v][0]}">${x.v}</span>${x.wx?' <span class="badge pr" title="Held back by the market-weather filter">weather</span>':''}</td>
   <td class="n">${x.s>0?'+':''}${x.s.toFixed(2)}</td><td class="n">${x.up==null?'<span class="muted">-</span>':x.up+'%'}</td>
   <td>${x.rl}</td><td class="n">${x.r30.chance_down_25}%</td><td class="n">${x.r30.chance_up_25}%</td>
   <td class="n">${pct(x.f.change_7d)}</td><td class="n">${fmtP(x.p)}</td></tr>`).join('');
  $('#calls-more').style.display = c.length>f.limit ? '' : 'none';
  $$('tr.row', body).forEach(tr=>tr.onclick=()=>{
    const n=tr.nextElementSibling; if(n && n.classList.contains('det')){ n.remove(); return; }
    const x=c[+tr.dataset.i]; const d=document.createElement('tr'); d.className='det';
    d.innerHTML = `<td colspan="9"><div class="dwrap"><p style="margin:0 0 10px">${x.op||''}</p><div class="dgrid">
      <div><h4>Strategy votes</h4><div class="kv"><span class="up">Buy</span><span>${x.buy.map(nice).join(', ')||'none'}</span></div>
        <div class="kv"><span class="down">Sell</span><span>${x.sell.map(nice).join(', ')||'none'}</span></div>
        <div class="kv"><span>RSI</span><span>${x.f.rsi}</span></div><div class="kv"><span>Last 30 days</span><span>${pct(x.f.change_30d)}</span></div>
        <div class="kv"><span>Below 90-day high</span><span>${pct(x.f.below_90d_high)}</span></div></div>
      <div><h4>Next 7 days (simulated)</h4><div class="kv"><span>Ends higher</span><span>${x.r7.chance_up}%</span></div>
        <div class="kv"><span>Up more than 10%</span><span>${x.r7.chance_up_10}%</span></div>
        <div class="kv"><span>Down more than 10%</span><span>${x.r7.chance_down_10}%</span></div>
        <div class="kv"><span>Down more than 25%</span><span>${x.r7.chance_down_25}%</span></div></div>
      <div><h4>Next 30 days (simulated)</h4><div class="kv"><span>Ends higher</span><span>${x.r30.chance_up}%</span></div>
        <div class="kv"><span>Up more than 25%</span><span>${x.r30.chance_up_25}%</span></div>
        <div class="kv"><span>Down more than 25%</span><span>${x.r30.chance_down_25}%</span></div>
        <div class="kv"><span>Down more than 50%</span><span>${x.r30.chance_down_50}%</span></div>
        <div class="kv"><span>Bad case / good case</span><span>${pct(x.r30.worst_5pct,0)} / ${pct(x.r30.best_5pct,0)}</span></div></div></div></div></td>`;
    tr.after(d);
  });
}
if ($('#calls-body')) {
  $('#q').oninput = e => { f.q=e.target.value.trim().toLowerCase(); f.limit=60; draw(); };
  $$('.chip[data-v]').forEach(b=>b.onclick=()=>{ $$('.chip[data-v]').forEach(x=>x.classList.remove('on')); b.classList.add('on'); f.v=b.dataset.v; f.limit=60; draw(); });
  $('#risk').onchange = e => { f.r=e.target.value; draw(); };
  $('#sort').onchange = e => { const [k,d]=e.target.value.split(':'); f.sort=k; f.dir=+d; draw(); };
  $$('th[data-k]').forEach(th=>th.onclick=()=>{ f.dir = f.sort===th.dataset.k ? -f.dir : -1; f.sort=th.dataset.k; draw(); });
  $('#calls-more button').onclick = () => { f.limit += 100; draw(); };
  draw();
}
// ---- trade filters ----
$$('.chip[data-tk]').forEach(b=>b.onclick=()=>{ $$('.chip[data-tk]').forEach(x=>x.classList.remove('on')); b.classList.add('on');
  $$('#closed tr[data-kind]').forEach(tr=>tr.style.display = (b.dataset.tk==='all' || tr.dataset.kind===b.dataset.tk) ? '' : 'none'); });
"""


def verdict_badge(v):
    return badge(v, {"STRONG BUY": "sb", "BUY": "b", "HOLD": "h", "SELL": "s", "STRONG SELL": "ss"}.get(v, "h"))


def build_page():
    tr = load("training_report.json", {}) or {}
    dr = load("daytrade_report.json", {}) or {}
    j = load("trade_journal.json", {"trades": []}) or {"trades": []}
    an = load("latest_analysis.json", {}) or {}
    state = load("state.json", {}) or {}
    hype = load("hype_history.json", {}) or {}
    trades = j.get("trades", [])
    prices = live_prices()
    now = time.time() * 1000
    cost = trade_cost()

    if not an.get("coins") and state:  # older bot runs: only verdicts are known
        an = {"coins": [{"c": c, "p": 0, "v": v, "s": 0, "up": None, "rl": "-", "r7": {}, "r30": {},
                         "buy": [], "sell": [], "f": {}, "op": ""} for c, v in state.items()]}
    coins = an.get("coins", [])
    counts = {v: sum(1 for c in coins if c["v"] == v) for v in ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")}

    daily_on = (tr.get("exam") or {}).get("model_trusted")
    dt_edge = (dr.get("exam") or {}).get("signals")
    safety = (j.get("safety") or {}).get("paused")
    real, prac = trade_stats(trades, False, now=now), trade_stats(trades, True, now=now)
    review = tr.get("call_review") or {}
    wx_d = tr.get("market_filter") or {}
    wx_t = (dr.get("exam") or {}).get("market_filter") or {}
    open_t = [t for t in trades if t["status"] == "open"]

    # ---------- OVERVIEW: plain-English summary ----------
    total = sum(counts.values()) or 1
    mood = ("cautious - most coins are rated SELL" if counts["SELL"] + counts["STRONG SELL"] > total / 2 else
            "positive - many coins are rated BUY" if counts["BUY"] + counts["STRONG BUY"] > total / 3 else
            "mixed - most coins are HOLD")
    say = [f"<p>Right now the bot is <b>{mood}</b> "
           f"({counts['STRONG BUY'] + counts['BUY']} buy, {counts['HOLD']} hold, "
           f"{counts['SELL'] + counts['STRONG SELL']} sell out of {sum(counts.values())} coins).</p>"]
    say.append("<p>The <b>AI model is " + ("ON" if daily_on else "OFF") + "</b> - " +
               ("it passed its last exam, so it helps decide." if daily_on else
                "it couldn't beat a coin flip on its last exam, so only the strategies that worked "
                "recently are deciding.") + "</p>")
    say.append("<p><b>Day-trade signals are " + ("ON" if dt_edge and not safety else "OFF") + "</b> - " +
               ("the last training found a profitable confidence level." if dt_edge and not safety else
                "paused by the safety switch." if safety else
                "no profitable setup was found after fees, so it only practice-trades (no alerts).") + "</p>")
    if review.get("reviewed"):
        say.append(f"<p>Of its past calls checked 7 days later, <b>{review['overall_right_pct']}% were right</b> "
                   f"({review['reviewed']} calls). See <a href='#track' data-go='track'>Track record</a> for the "
                   f"stricter 'beat Bitcoin' test.</p>")
    overview = (f"<div class='say'>{''.join(say)}</div><div class='stats'>"
                + stat("Daily AI model", "ON" if daily_on else "OFF", f"exam score {(tr.get('exam') or {}).get('auc', '-')}",
                       "up" if daily_on else "warn")
                + stat("Day-trade signals", "ON" if dt_edge and not safety else "OFF",
                       "safety switch ON" if safety else "edge found" if dt_edge else "no edge yet",
                       "up" if dt_edge and not safety else "warn")
                + stat("Open paper trades", len(open_t),
                       f"{sum(1 for t in open_t if t.get('practice'))} practice")
                + stat("Past calls right", f"{review['overall_right_pct']}%" if review.get("reviewed") else "-",
                       f"{review.get('reviewed', 0)} checked, {review.get('waiting_for_7_days', 0)} waiting")
                + "</div><h3>Today's calls</h3><div class='stats'>"
                + "".join(stat(verdict_badge(k), n) for k, n in counts.items())
                + "</div>")
    top_buys = sorted((c for c in coins if "BUY" in c["v"]), key=lambda c: -c["s"])[:8]
    if top_buys:
        overview += ("<h3>Strongest buy-rated coins</h3><div class='tw'><table><tr><th>Coin</th><th>Verdict</th>"
                     "<th class='n'>Score</th><th>Risk</th><th class='n'>Price</th></tr>" + "".join(
                         f"<tr><td><b>{esc(c['c'])}</b></td><td>{verdict_badge(c['v'])}</td><td class='n'>{c['s']:+.2f}</td>"
                         f"<td>{esc(c['rl'])}</td><td class='n'>${c['p']:.6g}</td></tr>" for c in top_buys)
                     + "</table></div><p class='hint'>Full list with search, filters and details: "
                       "<a href='#calls' data-go='calls'>Calls</a>.</p>")
    for label, w in (("Day trading", wx_t), ("Daily calls", wx_d)):
        if w:
            overview += (f"<p class='hint'>Market-weather filter ({label}): "
                         f"<b class='blue'>{'ON' if w.get('enabled') else 'OFF'}</b> - {esc(w.get('why', ''))}</p>")

    # ---------- CALLS ----------
    calls = f"""
<div class="stats">{''.join(stat(verdict_badge(k), n) for k, n in counts.items())}</div>
<div class="tools"><input type="search" id="q" placeholder="Search a coin, e.g. SOL">
<select id="risk" aria-label="Risk"><option value="ALL">Any risk</option><option>LOW</option><option>MEDIUM</option>
<option>HIGH</option><option>EXTREME</option></select>
<select id="sort" aria-label="Sort"><option value="s:-1">Most bullish first</option><option value="s:1">Most bearish first</option>
<option value="d25:1">Lowest crash risk</option><option value="u25:-1">Biggest upside</option>
<option value="ch7:-1">Best last 7 days</option><option value="c:1">A to Z</option></select></div>
<div class="tools">{''.join(f'<button class="chip{" on" if v == "ALL" else ""}" data-v="{v}">{lbl}</button>'
    for v, lbl in (("ALL", "All"), ("BUYS", "All buys"), ("STRONG BUY", "Strong buy"), ("BUY", "Buy"), ("HOLD", "Hold"),
                   ("SELL", "Sell"), ("STRONG SELL", "Strong sell"), ("SELLS", "All sells")))}
<span class="muted" id="calls-count" style="margin-left:auto;font-size:13px"></span></div>
<div class="tw"><table><thead><tr><th data-k="c">Coin</th><th>Verdict</th><th class="n" data-k="s">Score</th>
<th class="n" data-k="up">Chance up 7d</th><th data-k="rl">Risk</th><th class="n" data-k="d25">Crash risk 30d</th>
<th class="n" data-k="u25">+25% chance 30d</th><th class="n" data-k="ch7">Last 7d</th><th class="n" data-k="p">Price</th></tr></thead>
<tbody id="calls-body"></tbody></table></div>
<div class="more" id="calls-more"><button>Show more</button></div>
<p class="hint">Tap a coin for its opinion, strategy votes and full risk table. Click a column title to sort.
Crash risk = chance of falling more than 25% within 30 days (simulated).</p>"""
    if an.get("time"):
        calls = (f"<p class='intro'>From the latest analyze run (<span data-ago='{an['time']}'></span>). "
                 f"Every coin the bot rated, with its verdict and risk.</p>") + calls

    # ---------- DAY TRADING ----------
    dt = "<div class='stats'>" + stat(
        "Real signals (30 days)", f"{real['n']} trades" if real else "none yet",
        f"{real['win']:.0f}% won · {real['avg']:+.2f}% avg" if real else "only sent when there's an edge") + stat(
        "Practice trades (30 days)", f"{prac['n']} trades" if prac else "none yet",
        f"{prac['win']:.0f}% won · {prac['avg']:+.2f}% avg" if prac else "2 new every hour") + stat(
        "Signal threshold", f"{dt_edge['threshold']:.0%}" if dt_edge else "none",
        "minimum win chance to alert" if dt_edge else "no profitable setup found") + "</div>"
    dt += line_chart(
        [("Practice trades", "c1", curve(trades, True)), ("Real signals", "c2", curve(trades, False)),
         ("Just holding Bitcoin", "c3", btc_curve(trades, prices.get("BTC")))],
        "Running total of results after costs, vs. just holding Bitcoin", ref=((0, "0"),),
        note="Each line adds up every finished trade's result in %. Above the grey Bitcoin line = better than doing nothing.")
    rows_o = []
    for t in sorted(open_t, key=lambda t: -t["opened"]):
        p = prices.get(t["coin"])
        live = 100 * (p / t["entry"] - 1) - 100 * cost if p else None
        rows_o.append(f"<tr><td><b>{esc(t['coin'])}</b> {badge('practice', 'pr') if t.get('practice') else badge('signal', 'on')}</td>"
                      f"<td class='n'>{t['entry']:.6g}</td><td class='n'>{t['tp']:.6g}</td><td class='n'>{t['sl']:.6g}</td>"
                      f"<td class='n'>{signed(live)}</td><td>{ts(t['deadline'])}</td></tr>")
    dt += "<h3>Open trades</h3>" + (
        "<div class='tw'><table><tr><th>Coin</th><th class='n'>Bought at</th><th class='n'>Take-profit</th>"
        "<th class='n'>Stop-loss</th><th class='n'>Now (after costs)</th><th>Closes by (UTC)</th></tr>"
        + "".join(rows_o) + "</table></div>" if rows_o else "<div class='empty'>No open trades right now.</div>")
    closed = sorted((t for t in trades if t["status"] != "open"), key=lambda t: -t.get("closed", t["opened"]))[:60]
    if closed:
        dt += ("<h3>Finished trades</h3><div class='tools'>" + "".join(
            f"<button class='chip{' on' if k == 'all' else ''}' data-tk='{k}'>{lbl}</button>"
            for k, lbl in (("all", "All"), ("real", "Real signals"), ("practice", "Practice"))) + "</div>"
               "<div class='tw'><table id='closed'><tr><th>Coin</th><th>Result</th><th class='n'>After costs</th>"
               "<th>Opened (UTC)</th></tr>" + "".join(
                   f"<tr data-kind='{'practice' if t.get('practice') else 'real'}'><td><b>{esc(t['coin'])}</b> "
                   f"{badge('practice', 'pr') if t.get('practice') else badge('signal', 'on')}</td>"
                   f"<td>{esc({'won': 'hit take-profit', 'lost': 'hit stop-loss', 'timeout': '12h time limit'}.get(t['status'], t['status']))}</td>"
                   f"<td class='n'>{signed(100 * t['net'])}</td><td>{ts(t['opened'])}</td></tr>" for t in closed)
               + "</table></div>")
    if wx_t:
        dt += (f"<p class='hint'>Market-weather filter: <b class='blue'>{'ON' if wx_t.get('enabled') else 'OFF'}</b>"
               f" - {esc(wx_t.get('why', ''))}</p>")

    # ---------- HYPE ----------
    last = (hype.get("runs") or [{}])[-1] if hype.get("runs") else {}
    hrows = "".join(f"<tr><td><b>{esc(h['coin'])}</b></td><td class='n'>{h['hype_score']}</td>"
                    f"<td>{badge(h['stage'], 'on' if h['stage'] in ('EARLY', 'BUILDING') else 'h')}</td>"
                    f"<td class='n'>${h['price']:.6g}</td></tr>" for h in last.get("top", []))
    hype_html = ((f"<p class='intro'>From the latest hype scan (<span data-ago='{last['time'] * 1000}'></span>): "
                  "coins people are talking about on Reddit, Bluesky, 4chan /biz/ and CoinGecko trending.</p>"
                  "<div class='tw'><table><tr><th>Coin</th><th class='n'>Hype (0-100)</th><th>Stage</th>"
                  "<th class='n'>Price</th></tr>" + hrows + "</table></div>"
                  "<p class='hint'>EARLY = buzz before the price moved (the only stage where hype sometimes pays). "
                  "LATE = already pumped. Most hyped coins are lower a month later.</p>")
                 if hrows else "<div class='empty'>No hype scan yet.</div>")

    # ---------- TRACK RECORD ----------
    tr_rows = []
    for k, v in sorted((review.get("by_call") or {}).items(), key=lambda kv: -kv[1]["calls"]):
        kind, call = k.split(":", 1)
        beat = (f"{v['beat_btc']}/{v['beat_btc_of']} ({100 * v['beat_btc'] / v['beat_btc_of']:.0f}%)"
                if v.get("beat_btc_of") else "-")
        tr_rows.append(f"<tr><td>{esc(kind)}</td><td>{badge(call, 'pr' if 'PRACTICE' in call else 'h')}</td>"
                       f"<td class='n'>{v['calls']}</td><td class='n'>{v['right_pct']}%</td>"
                       f"<td class='n'>{signed(v['avg_7d_return_pct'])}</td><td class='n'>{signed(v.get('btc_same_weeks_pct'))}</td>"
                       f"<td class='n'>{beat}</td></tr>")
    bars = "".join(
        f"<div class='r'><span>{esc(k.split(':', 1)[1])}</span><div class='t'><div class='f' style='width:{v['right_pct']}%'></div></div>"
        f"<span class='n'>{v['right_pct']:.0f}%</span></div>"
        for k, v in sorted((review.get("by_call") or {}).items(), key=lambda kv: -kv[1]["calls"])[:8])
    track = (("<div class='stats'>" + stat("Calls checked", review.get("reviewed", 0)) +
              stat("Right overall", f"{review['overall_right_pct']}%" if review.get("overall_right_pct") is not None else "-")
              + stat("Waiting for 7 days", review.get("waiting_for_7_days", 0)) + "</div>"
              + (f"<div class='card bars'>{bars}</div>" if bars else "")
              + "<h3>Every kind of call</h3><div class='tw'><table><tr><th>Mode</th><th>Call</th><th class='n'>Calls</th>"
                "<th class='n'>Right</th><th class='n'>Avg 7 days after</th><th class='n'>Bitcoin same weeks</th>"
                "<th class='n'>Beat Bitcoin</th></tr>" + "".join(tr_rows) + "</table></div>"
              + "<p class='hint'>Right = buy calls went up, sell calls went down. <b class='blue'>Beat Bitcoin</b> is the "
                "honest test: if sell calls are only 'right' because the whole market fell, they don't beat Bitcoin.</p>")
             if tr_rows else "<div class='empty'>No calls are 7 days old yet - the first ones show up a week after "
                             "they were made.</div>")

    # ---------- MODELS ----------
    weights = tr.get("strategy_weights") or {}
    wbars = "".join(
        f"<div class='r'><span>{esc(k.replace('_', ' '))}</span><div class='t'><div class='f' style='width:{100 * w:.0f}%'></div></div>"
        f"<span class='n'>{100 * w:.0f}%</span></div>" for k, w in sorted(weights.items(), key=lambda kv: -kv[1]) if w > 0)
    models = (line_chart(
        [("Daily model", "c1", [(date_ms(h["date"]), h["auc"]) for h in tr.get("history", []) if date_ms(h["date"])]),
         ("Day-trade model", "c2", [(date_ms(h["date"]), h["auc"]) for h in dr.get("history", []) if date_ms(h["date"])])],
        "Exam score after each training", ref=((0.5, "coin flip"), (0.53, "switch-on level")), pct=False,
        note="The exam uses recent data the model never trained on. Above the 0.53 line the daily model is used; "
             "below it, only the strategies decide.")
              + "<h3>Which strategies the bot trusts right now</h3>"
              + (f"<div class='card bars'>{wbars}</div>" if wbars else "<div class='empty'>No strategy made money recently.</div>")
              + "<p class='hint'>Trust is based on the last ~8 months: a strategy only gets a vote if its typical trade "
                "made money.</p><div class='stats'>"
              + stat("Daily model settings", esc((tr.get("exam") or {}).get("model", "-")), f"trained {esc(tr.get('trained_at', '-'))[:16]}", "sm")
              + stat("Day-trade model settings", esc((dr.get("exam") or {}).get("model", "-")), f"trained {esc(dr.get('trained_at', '-'))}", "sm")
              + "</div>")

    guide = ("<p class='intro'>What everything on this page means, in plain words.</p><dl class='g'>"
             + "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in GUIDE) + "</dl>"
             "<p class='hint'>Everything here is paper trading. Treat calls as 'worth a look', never 'buy now'. "
             "Not financial advice.</p>")

    tabs = [("overview", "Overview", "", overview),
            ("calls", "Calls", "", calls),
            ("trading", "Day trading", "Short trades with a take-profit, a stop-loss and a 12-hour limit. "
                                       "Paper trades only.", dt),
            ("hype", "Hype", "", hype_html),
            ("track", "Track record", "Every call the bot made, checked 7 days later.", track),
            ("models", "Models", "How the AI is doing and what it trusts.", models),
            ("guide", "Guide", "", guide)]
    nav = "".join(f'<button data-t="{t}">{name}</button>' for t, name, _, _ in tabs)
    body = "".join(section(t, name, intro, b) for t, name, intro, b in tabs)
    data = json.dumps({"analysis": {"coins": coins}}, separators=(",", ":")).replace("</", "<\\/")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#000000">
<title>Crypto Bot Dashboard</title><style>{CSS}</style></head><body>
<header><div class="bar"><div class="logo">Crypto<b>Bot</b></div>
<div class="upd">Updated <span data-ago="{int(now)}"></span> · paper trading only</div></div>
<nav aria-label="Sections">{nav}</nav></header>
<main>{body}</main>
<footer>Built automatically from the bot's files · paper trading only · not financial advice</footer>
<script id="data" type="application/json">{data}</script><script>{JS}</script></body></html>"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page)
    (DOCS / ".nojekyll").write_text("")
    print(f"Dashboard written: docs/index.html ({len(page) // 1024} KB, {len(coins)} coins)")
    return {"real": real, "prac": prac, "review": review, "daily_on": daily_on, "dt_edge": dt_edge,
            "safety": safety, "trades": trades, "prices": prices}


# ------------------------------------------------------------- weekly push ---
def weekly_due(force):
    st = json.loads(SUMMARY_STATE.read_text()) if SUMMARY_STATE.exists() else {}
    now = datetime.now(timezone.utc)
    last = st.get("last_sent", 0)
    sunday_evening = now.weekday() == 6 and now.hour >= 17
    return force or (sunday_evening and time.time() - last > 3 * 86400) or (last and time.time() - last > 8 * 86400)


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
