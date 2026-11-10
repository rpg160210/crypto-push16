# Full setup guide (about 20 minutes, no coding)

You'll set up 4 things: the **ntfy** app (notifications), a **GitHub** account (runs the bot
for free in the cloud), a **CoinGecko** key (price data) and, optionally, an **Anthropic** key
(Claude writes the opinions).

---

## Step 1 - Notifications on your phone (ntfy)
1. Install **ntfy** from the App Store or Google Play (free, no account needed).
2. Open it → tap **+** (Subscribe to topic).
3. Make up a random topic name nobody could guess, e.g. `crypto-alerts-k39fz2x7`.
   Anyone who knows the name can read your alerts, so don't use something like `crypto`.
4. Leave the server as `ntfy.sh` → **Subscribe**. Write the topic name down.
5. (iPhone) Allow notifications when it asks.

## Step 2 - CoinGecko key (free price data)
1. Go to **coingecko.com/en/api/pricing** → choose the free **Demo** plan → sign up.
2. In the developer dashboard click **+ Add New Key** and copy the key (starts with `CG-`).
   Without the key the first training takes 2+ hours and may hit rate limits.

## Step 3 - Anthropic key (optional, ~a few cents per run)
Skip this if you're fine with the built-in opinions.
1. Go to **console.anthropic.com** → sign up → **Billing**, add a few dollars of credit.
2. **API Keys** → **Create Key** → copy it (starts with `sk-ant-`). You only see it once.
3. Tip: set a monthly spend limit under **Limits**.

## Step 4 - Create the GitHub repository
1. Sign up at **github.com** (free).
2. Click **+** (top right) → **New repository**.
3. Name it `crypto-ai-analyst`. Choose **Private**. Tick **Add a README file**. Click **Create repository**.

## Step 5 - Upload the bot's files
1. Unzip `crypto-ai-analyst.zip` on your computer.
2. In your repo click **Add file → Upload files**. Drag in these 6 files:
   `analyze.py`, `core.py`, `train.py`, `requirements.txt`, `README.md`, `SETUP.md`.
   Click **Commit changes**. (It's fine if it replaces the README.)
3. Now the two automation files. Folders starting with a dot are often hidden on
   computers, so create them by hand:
   - Click **Add file → Create new file**.
   - In the name box type exactly: `.github/workflows/train.yml`
     (typing `/` makes the folders automatically).
   - Open `train.yml` from the unzipped `.github/workflows` folder in a text editor
     (Notepad / TextEdit), copy everything, paste it into GitHub → **Commit changes**.
     If you can't see the folder: on Mac press **Cmd+Shift+.** in Finder; on Windows,
     View → Show → Hidden items.
   - Do the same for `.github/workflows/analyze.yml`.

## Step 6 - Add your keys (secrets)
In the repo: **Settings → Secrets and variables → Actions**.

**Secrets** tab → **New repository secret**, add each one:

| Name | Value |
|---|---|
| `NTFY_TOPIC` | your topic from Step 1, e.g. `crypto-alerts-k39fz2x7` |
| `COINGECKO_API_KEY` | key from Step 2 |
| `ANTHROPIC_API_KEY` | key from Step 3 (optional) |

**Variables** tab → **New repository variable**:

| Name | Value |
|---|---|
| `COINS` | your watchlist, comma-separated, no spaces, e.g. `bitcoin,ethereum,solana,dogecoin,pepe` |

**Want every coin on Binance?** Set `COINS` to just `binance`. The bot then pulls the full
Binance list (~400 coins) and its prices straight from Binance, and sends you one summary
alert per run plus details for the 3 strongest new calls instead of hundreds of alerts.

Otherwise, names in `COINS` are **CoinGecko IDs**, not tickers. Find one by opening the coin on
coingecko.com: the ID is in the web address, e.g. `coingecko.com/en/coins/ripple` → `ripple` (not `xrp`).

## Step 7 - Train the AI (once, then automatic every Sunday)
1. Open the **Actions** tab. If asked, click **I understand my workflows, go ahead and enable them**.
2. Click **train** on the left → **Run workflow** → **Run workflow**.
3. Wait ~40-60 minutes (it downloads 1000 coins). Green check = done.
4. Back on the **Code** tab you'll now see `model.joblib` and `training_report.json`.
   Open `training_report.json` and look at `"exam"`:
   - `accuracy` vs `accuracy_if_you_always_guessed_the_majority` - if accuracy isn't
     clearly higher, the model has no real edge right now. Trust it less.
   - `strategy_backtest_test_period` shows which strategies actually worked lately.

## Step 8 - Turn on the analyst
1. **Actions → analyze → Run workflow**.
2. Within a couple of minutes your phone should get one notification per coin.
3. Click the finished run → **analyze** → expand **Run python analyze.py --notify** to see
   the full report with all the risk percentages.
4. From now on it runs every 8 hours and only notifies you when a verdict changes.

---

## Changing things later
- **Watchlist:** edit the `COINS` variable (Step 6).
- **Get every coin every time** (not just changes): edit `.github/workflows/analyze.yml`,
  change `python analyze.py --notify` to `python analyze.py --notify-all`.
- **How often:** in `analyze.yml` change `"23 */8 * * *"` (every 8 hours) - e.g. `"23 */4 * * *"` = every 4 hours.
  With `COINS=binance` in a private repo, keep it at 8 hours or you'll run out of free GitHub minutes.
- **Fewer coins for training (faster):** add a variable `TRAIN_COINS` = `300`.

## Troubleshooting
| Problem | Fix |
|---|---|
| analyze fails with "No trained model yet" | Run **train** first and wait for the green check |
| No notifications | Check the `NTFY_TOPIC` secret matches the app exactly (case-sensitive). Check phone notification settings for ntfy |
| A coin is skipped | Wrong ID - use the CoinGecko ID from the web address |
| "Permission denied" / push failed | Settings → Actions → General → Workflow permissions → **Read and write** → Save |
| train fails with rate limit | Make sure `COINGECKO_API_KEY` is set, then run it again (it keeps what it already downloaded) |
| Bot stopped running after ~2 months | GitHub pauses schedules in quiet repos. Actions tab → pick the workflow → **Enable workflow** |

## Cost
- ntfy, CoinGecko Demo: free.
- GitHub: private repos get 2,000 free Action minutes/month; this bot uses roughly 500-700.
  (Public repos are unlimited, but then anyone could see your watchlist.)
- Claude: optional, typically a few cents per run → a few dollars a month at every 4 hours.

**Not financial advice.** Treat alerts as "go have a look", never as "buy now".
