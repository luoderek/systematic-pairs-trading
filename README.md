# Systematic Pairs Trading & Optimisation Engine

[![Launch App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://luoderek-pairs-trading.streamlit.app)

A market-neutral statistical arbitrage backtester with an interactive dashboard. Inputting any two stock tickers, it checks whether they're actually cointegrated, builds a hedged spread between them, and trades the spread when it gets stretched too far from its mean.

The maths sits in `strategy_engine.py`. The Streamlit app in `app.py` is a front end over it, so the engine can be imported and used on its own.

## The concept

Pairs trading looks for two stocks tied to the same underlying economics, like AMD and Analog Devices, whose prices tend to move together. When one runs ahead of the other, you short the expensive one and buy the cheap one, betting the gap closes. If it does, you make money whether the market went up or down that month.

The correlation between two stocks isn't enough. Two tech names will both rise in a bull market without the gap between them being predictable at all. What is instead needed is that the stocks are cointegrated. That is, the gap itself has to be stationary, so it has a mean it keeps reverting to.

## Building the spread

The app runs an Engle-Granger test on log prices to check the spread is stationary before you trade it. A p-value under 0.05 is a pass.

The hedge ratio isn't fixed. It comes from a rolling OLS regression over a trailing window, so it updates as the relationship between the two stocks drifts. The spread is the regression residual, intercept included:

$$\text{Spread}_t = \ln(Y_t) - \left(\alpha_t + \beta_t \ln(X_t)\right)$$

That gets normalised into a rolling z-score, so the trading rules work on standard deviations instead of price levels and don't need retuning for every new pair:

$$Z_t = \frac{\text{Spread}_t - \mu_t}{\sigma_t}$$

The app also reports the spread's half-life, worked out from an Ornstein-Uhlenbeck fit. It's roughly how many days the spread takes to close half the gap back to zero, and it tells you whether your z-score window is sensible. A 30 day window on a spread with a 90 day half-life is normalising against the wrong mean most of the time, and the sidebar will say so.

## Trading rules

A state machine holds one of three positions: long the spread, short it, or flat.

- **Open** when the z-score goes past the entry boundary, but only if it hasn't already blown through the stop. A spread that gaps straight past the stop gets skipped.
- **Close at target** when the z-score comes back to the exit level. Zero waits for the full mean reversion; anything above that takes profit early.
- **Stop out** if the z-score keeps going the wrong way past the stop loss.
- **Cooldown** optionally blocks new entries for a few days after a stop, so the engine doesn't keep jumping back into a spread that's falling apart.

## Working out returns

**Position sizing.** One unit of position is 1 notional of Y against |β| of X, so the capital tied up is (1 + |β|), not 1. Returns get divided by that. Without this, the results look better than they actually are any time β is above 1.

**Simple returns, not log.** Log returns add across time but not across positions, so the two legs are combined in simple-return space and compounded.

**Costs.** Commission plus half the bid-ask, charged in basis points on the traded notional of both legs. It's worked out from the daily change in leg weights, which picks up the small rebalancing cost as β drifts, not only the entries and exits. Costs matter enough here to change which entry threshold looks best, so the optimiser runs on net returns.

**Execution timing.** Default is a T+0 fill: the signal from the close of day T is filled at that close and earns the move into T+1. Nothing looks forward, but it does assume you can trade the closing print. There's a T+1 option in the sidebar that waits a full day.

**Warm-up.** The rolling windows need about a year of data before they produce a signal. Those days are dropped from the metrics instead of being left in as zeros, which would drag the volatility down and make the track record look longer than it is.

## Choosing the entry threshold

The optimiser sweeps entry boundaries from 1.0 to 3.0 standard deviations and scores each one on total return, Sortino, Sharpe or Calmar.

Sortino uses a proper downside deviation, meaning the root mean square of everything below target divided by the total number of days:

$$\sigma_d = \sqrt{\frac{1}{N}\sum_{t=1}^{N} \min(r_t - \tau,\ 0)^2}\ \cdot \sqrt{252}$$

The sweep only scores the first 70% of the data. The rest is held back so the dashboard can show a result the optimiser never saw. Picking a threshold on the full history and then reporting on that same history isn't really a result, it's just the best fit. The app shows both windows side by side and warns you if the out-of-sample Sharpe drops off badly, which usually means the threshold is overfitted.

## The dashboard

- Cointegration check on both the in-sample window and the trailing year, so you can see if the relationship has broken down recently
- Equity curve with the held-out period shaded
- Z-score oscillator with the entry and stop levels drawn on
- Metrics table split into in-sample, out-of-sample and full period
- Threshold sweep chart showing how each objective scores the grid
- Trade log with entry and exit dates, holding periods, per-trade returns and costs
- Rolling hedge ratio, worth a look because a β that jumps around means the relationship isn't stable no matter what the p-value says

## Stack

Python, with pandas and NumPy for the data handling, statsmodels for the rolling regression and the cointegration test, and Streamlit and Plotly for the interface.

## Running it

```bash
git clone https://github.com/luoderek/systematic-pairs-trading.git
cd systematic-pairs-trading
pip install -r requirements.txt
streamlit run app.py
```

Data comes from yfinance, so you need a connection the first time you load a pair. Results are cached for an hour after that.

## Files

```
app.py               Streamlit dashboard
strategy_engine.py   spread construction, state machine, P&L, metrics, optimiser
requirements.txt
```

## Limitations

- No borrow cost on the short leg, and no check that the stock is shortable at all. Fine for large caps, would matter a lot for anything hard to borrow.
- Close-to-close fills only. No intraday slippage, no market impact, no partial fills.
- The hedge ratio comes from regressing Y on X. Swap them around and you get a different β, so neither one is really the cointegrating vector. Johansen would handle this properly.
- Sweeping 21 thresholds means the best in-sample score is partly luck. The held-out window helps but doesn't correct for it.