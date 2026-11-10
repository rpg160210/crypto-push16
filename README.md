# Crypto AI Analyst 🧠📊

A bot that learns from ~1000 cryptocurrencies, combines 8 classic trading strategies with a
machine-learning model, gives each coin a **verdict with an opinion**, and shows the **risk in
percentages** from 20,000 simulated futures. It pushes verdict changes to your phone.

> ⚠️ **Not financial advice.** Nothing can check "every possible outcome" or predict crypto
> reliably. Expect the model to be right a bit more often than a coin flip on real data.
> It will often be wrong. Only risk money you can afford to lose.

## What it does
| Step | What happens |
|---|---|
| **Strategies** | Trend following (EMA20/50), momentum, RSI reversion, MACD, Bollinger reversion, Turtle 20-day breakout, volume breakout, buy-the-dip-in-uptrend |
| **Training** (`train.py`) | Downloads 1 year of daily prices for the top ~1000 coins (stablecoins skipped). Back-tests every strategy, then trains and compares 3 ML models on ~300k coin-days to predict "higher in 7 days?". The latest ~3 months are hidden and used as a final exam, so the reported accuracy is on unseen data. |
| **Verdict** | 60% model probability + 40% strategy vote (strategies that worked better in back-tests get more weight) → STRONG BUY / BUY / HOLD / SELL / STRONG SELL |
| **Opinion** | Blunt built-in opinion, or, with an Anthropic API key, Claude writes the take + biggest risk |
| **Risk %** | 20,000 Monte Carlo futures built from the coin's own past daily swings (tilted toward the model's 7-day view): chance of ending up/down, ±10%, ±25%, −50%, a 20% drop along the way, typical range, worst/best 5% |
| **Alerts** | ntfy push when a coin's verdict changes |

## Setup (free)
1. Install the **ntfy** app on your phone and subscribe to a random topic name (e.g. `crypto-k39fz2`).
2. Create a GitHub repo (private is fine) and upload these files.
3. **Settings → Secrets and variables → Actions**:
   - Secret `NTFY_TOPIC` (required)
   - Secret `COINGECKO_API_KEY` (strongly recommended: free "Demo" key from coingecko.com. Without it the first training run takes 2+ hours)
   - Secret `ANTHROPIC_API_KEY` (optional: Claude writes the opinions)
   - Variable `COINS` = your watchlist, e.g. `bitcoin,ethereum,solana,pepe` (CoinGecko IDs), or just `binance` for every coin on Binance
4. **Actions → train → Run workflow**. Wait for it to finish (it commits `model.joblib`).
5. **Actions → analyze → Run workflow**. From then on it runs every 8 hours.

Check `training_report.json` for the model's real exam score and the strategy scoreboard.

## Run locally
```
pip install -r requirements.txt
python train.py --demo                  # 1 minute, fake data, just to see it work
python analyze.py bitcoin solana --demo
python train.py                         # the real thing (1000 coins)
python analyze.py bitcoin solana pepe
```
