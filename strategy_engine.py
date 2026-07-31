import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import statsmodels.tsa.stattools as ts

TRADING_DAYS = 252
EPS = 1e-12


# ----------------------------------------------------------------------------
# Structural tests
# ----------------------------------------------------------------------------

def test_cointegration(series_y, series_x):
    """
    Engle-Granger two-step cointegration test on log prices.

    Returns the p-value, or NaN if there is not enough clean overlapping data
    for the test to be meaningful.
    """
    frame = pd.concat([series_y, series_x], axis=1).dropna()
    if len(frame) < 30 or frame.iloc[:, 0].std() == 0 or frame.iloc[:, 1].std() == 0:
        return float("nan")
    try:
        _, pvalue, _ = ts.coint(frame.iloc[:, 0], frame.iloc[:, 1])
    except Exception:
        return float("nan")
    return float(pvalue)


def half_life(spread):
    """
    Ornstein-Uhlenbeck half-life of mean reversion, in trading days.

    Regresses the daily change in the spread on its lagged level. A positive
    kappa means no mean reversion.
    """
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return float("nan")
    lagged = s.shift(1)
    delta = s.diff()
    frame = pd.concat([delta, lagged], axis=1).dropna()
    frame.columns = ["delta", "lagged"]
    model = sm.OLS(frame["delta"], sm.add_constant(frame["lagged"])).fit()
    kappa = float(model.params.iloc[1])
    if kappa >= 0:
        return float("inf")
    return float(-np.log(2) / kappa)


# ----------------------------------------------------------------------------
# Spread construction (threshold-independent)
# ----------------------------------------------------------------------------

def compute_spread(log_data, x_ticker, y_ticker, beta_lookback=252, signal_lookback=30):
    """
    Rolling-OLS hedge ratio, dynamic spread, and normalised z-score.

    Returns a DataFrame indexed like `log_data` with columns:
        alpha, beta, spread, z
    """
    if x_ticker == y_ticker:
        raise ValueError("Asset X and Asset Y must be different tickers.")

    frame = log_data[[y_ticker, x_ticker]].dropna()
    min_rows = beta_lookback + signal_lookback + 5
    if len(frame) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} overlapping observations for a "
            f"{beta_lookback}-day beta window and {signal_lookback}-day z-score "
            f"window; got {len(frame)}."
        )

    y_log = frame[y_ticker]
    x_log = sm.add_constant(frame[x_ticker])

    fit = RollingOLS(y_log, x_log, window=beta_lookback).fit()
    alpha = fit.params["const"]
    beta = fit.params[x_ticker]

    spread = y_log - (alpha + beta * frame[x_ticker])
    mean = spread.rolling(window=signal_lookback).mean()
    std = spread.rolling(window=signal_lookback).std()

    # Guard against a degenerate flat window producing an infinite z-score.
    std = std.where(std > EPS)
    z = (spread - mean) / std

    return pd.DataFrame(
        {"alpha": alpha, "beta": beta, "spread": spread, "z": z},
        index=frame.index,
    )


# ----------------------------------------------------------------------------
# Path-dependent state machine
# ----------------------------------------------------------------------------

def run_state_machine(z, entry_thresh, exit_thresh=0.0, stop_loss=3.0, cooldown=0):
    """
    Path-dependent position state machine over the z-score oscillator.

    States are -1 (short spread), 0 (flat), +1 (long spread). Entry requires
    the z-score to be beyond `entry_thresh` but not already past `stop_loss`,
    so a spread that gaps straight through the stop is never entered.

    `cooldown` bars re-entry for N days after a stop-out, which stops the engine
    repeatedly re-arming into a spread that is actively breaking down.

    Returns (positions, exit_reasons); exit_reasons records 'target' or 'stop'
    on the bar a position is closed.
    """
    if entry_thresh >= stop_loss:
        raise ValueError(
            f"Entry threshold ({entry_thresh:.2f}) must be strictly below the "
            f"stop loss ({stop_loss:.2f}); otherwise no entry is ever possible."
        )

    z_array = np.asarray(z, dtype=float)
    n = len(z_array)
    positions = np.zeros(n)
    reasons = np.empty(n, dtype=object)

    state = 0
    blocked_until = -1

    for i in range(n):
        value = z_array[i]

        if np.isnan(value):
            state = 0
            positions[i] = 0.0
            continue

        if state == 0:
            if i >= blocked_until:
                if entry_thresh <= value < stop_loss:
                    state = -1            # spread rich -> short the spread
                elif -entry_thresh >= value > -stop_loss:
                    state = 1             # spread cheap -> long the spread

        elif state == 1:
            if value <= -stop_loss:
                state = 0
                reasons[i] = "stop"
                blocked_until = i + 1 + cooldown
            elif value >= -exit_thresh:
                state = 0
                reasons[i] = "target"

        elif state == -1:
            if value >= stop_loss:
                state = 0
                reasons[i] = "stop"
                blocked_until = i + 1 + cooldown
            elif value <= exit_thresh:
                state = 0
                reasons[i] = "target"

        positions[i] = state

    index = z.index if isinstance(z, pd.Series) else None
    return (
        pd.Series(positions, index=index, name="position"),
        pd.Series(reasons, index=index, name="exit_reason"),
    )


def generate_signals(log_data, x_ticker, y_ticker, beta_lookback=252,
                     signal_lookback=30, entry_thresh=2.0, exit_thresh=0.0,
                     stop_loss=3.0, cooldown=0):
    """
    Convenience wrapper: build the spread and run the state machine in one call.

    Returns (spread_frame, positions, exit_reasons).
    """
    spread_frame = compute_spread(
        log_data, x_ticker, y_ticker, beta_lookback, signal_lookback
    )
    positions, reasons = run_state_machine(
        spread_frame["z"], entry_thresh, exit_thresh, stop_loss, cooldown
    )
    return spread_frame, positions, reasons


# ----------------------------------------------------------------------------
# Portfolio construction and P&L
# ----------------------------------------------------------------------------

def calculate_equity(log_data, spread_frame, positions, x_ticker, y_ticker,
                     cost_bps=5.0, execution_lag=1):
    """
    Convert signals into a capital-normalised, cost-aware net return stream.

    `cost_bps` is charged per unit of notional traded, per leg, and covers
    commission plus half the bid-ask spread.

    Returns (equity_curve, net_returns, book) where `equity_curve` is cumulative
    return in decimal and `book` carries the leg weights, turnover and costs.
    """
    if execution_lag < 1:
        raise ValueError(
            "execution_lag must be at least 1 to avoid lookahead bias.")

    index = spread_frame.index
    y_log = log_data[y_ticker].reindex(index)
    x_log = log_data[x_ticker].reindex(index)

    # Simple returns: legs cannot be aggregated additively in log space.
    ret_y = np.exp(y_log.diff()) - 1.0
    ret_x = np.exp(x_log.diff()) - 1.0

    held = positions.shift(execution_lag)
    beta_exec = spread_frame["beta"].shift(execution_lag)

    gross = 1.0 + beta_exec.abs()
    weight_y = (held / gross).fillna(0.0)
    weight_x = (-held * beta_exec / gross).fillna(0.0)

    gross_return = weight_y * ret_y + weight_x * ret_x

    # Turnover on both legs; captures rebalancing drag from a drifting beta,
    # not just the round trips.
    turnover = (weight_y.diff().abs() + weight_x.diff().abs()).fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)

    net_returns = (gross_return - costs).fillna(0.0)

    # Trim the warm-up. Leaving the forced zeros in deflates volatility and
    # inflates the apparent length of the track record.
    first = spread_frame["z"].first_valid_index()
    if first is not None:
        start = index.get_loc(first)
        net_returns = net_returns.iloc[start:]
        held = held.iloc[start:]
        turnover = turnover.iloc[start:]
        costs = costs.iloc[start:]
        weight_y = weight_y.iloc[start:]
        weight_x = weight_x.iloc[start:]

    equity_curve = (1.0 + net_returns).cumprod() - 1.0

    book = pd.DataFrame({
        "held": held,
        "weight_y": weight_y,
        "weight_x": weight_x,
        "turnover": turnover,
        "cost": costs,
        "net_return": net_returns,
    })

    return equity_curve, net_returns, book


# ----------------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------------

def downside_deviation(returns, target=0.0, periods=TRADING_DAYS):
    """
    Annualised downside deviation: root-mean-square shortfall below `target`,
    averaged over all periods.
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(periods))


def _empty_metrics(n=0):
    return {
        "Total Return": 0.0, "CAGR": 0.0, "Annualised Volatility": 0.0,
        "Sharpe Ratio": 0.0, "Sortino Ratio": 0.0, "Calmar Ratio": 0.0,
        "Max Drawdown": 0.0, "Daily Win Rate": 0.0, "Trades": 0,
        "Trade Win Rate": 0.0, "Avg Trade Return": 0.0,
        "Time In Market": 0.0, "Total Cost": 0.0, "Observations": int(n),
    }


def calculate_tearsheet(net_returns, book=None, risk_free=0.0, periods=TRADING_DAYS):
    """
    Risk-adjusted performance metrics from a stream of net simple returns.

    `risk_free` is an annualised decimal rate and is subtracted before the
    Sharpe and Sortino calculations, so both are excess-return ratios.
    """
    r = pd.Series(net_returns).dropna()

    if len(r) < 2 or float(r.abs().sum()) == 0.0:
        return _empty_metrics(len(r))

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = len(r) / periods
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    daily_rf = risk_free / periods
    excess = r - daily_rf

    volatility = float(r.std(ddof=1) * np.sqrt(periods))
    sharpe = float(excess.mean() / (excess.std(ddof=1) + EPS)
                   * np.sqrt(periods))

    dd_dev = downside_deviation(excess, target=0.0, periods=periods)
    sortino = float(excess.mean() * periods / (dd_dev + EPS))

    running_max = equity.cummax()
    max_drawdown = float(((equity - running_max) / running_max).min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    active = r[r != 0]
    daily_win_rate = float((active > 0).mean()) if len(active) else 0.0

    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Daily Win Rate": daily_win_rate,
        "Observations": int(len(r)),
        "Trades": 0,
        "Trade Win Rate": 0.0,
        "Avg Trade Return": 0.0,
        "Time In Market": 0.0,
        "Total Cost": 0.0,
    }

    if book is not None and len(book):
        trades = summarise_trades(book)
        metrics["Trades"] = int(len(trades))
        metrics["Time In Market"] = float((book["held"].fillna(0) != 0).mean())
        metrics["Total Cost"] = float(book["cost"].sum())
        if len(trades):
            metrics["Trade Win Rate"] = float((trades["return"] > 0).mean())
            metrics["Avg Trade Return"] = float(trades["return"].mean())

    return metrics


def summarise_trades(book):
    """
    Collapse the daily book into round-trip trades.

    A trade is a maximal run of constant non-zero executed position.
    """
    columns = ["entry", "exit", "direction", "days", "return", "cost"]
    if book is None or len(book) == 0:
        return pd.DataFrame(columns=columns)

    held = book["held"].fillna(0.0).to_numpy()
    net = book["net_return"].to_numpy()
    cost = book["cost"].to_numpy()
    n = len(held)

    records = []
    i = 0
    while i < n:
        if held[i] == 0:
            i += 1
            continue
        start = i
        while i + 1 < n and held[i + 1] == held[start]:
            i += 1
        end = i

        tail = end + 1 if (end + 1 < n and held[end + 1] == 0) else None
        block_net = net[start:end + 1]
        block_cost = cost[start:end + 1].sum()
        compounded = float(np.prod(1.0 + block_net))
        if tail is not None:
            compounded *= (1.0 + net[tail])
            block_cost += cost[tail]

        records.append({
            "entry": book.index[start],
            "exit": book.index[tail if tail is not None else end],
            "direction": "long spread" if held[start] > 0 else "short spread",
            "days": int(end - start + 1),
            "return": compounded - 1.0,
            "cost": float(block_cost),
        })
        i = end + 1

    return pd.DataFrame(records, columns=columns)


# ----------------------------------------------------------------------------
# Threshold optimisation
# ----------------------------------------------------------------------------

OBJECTIVES = {
    "max profit": "Total Return",
    "sortino": "Sortino Ratio",
    "sharpe": "Sharpe Ratio",
    "calmar": "Calmar Ratio",
}


def run_strategy_factory(log_data, x_ticker, y_ticker, objective="sortino",
                         beta_lookback=252, signal_lookback=30, exit_thresh=0.0,
                         stop_loss=3.0, cooldown=0, cost_bps=5.0,
                         execution_lag=1, risk_free=0.0, split_frac=0.70,
                         grid=None, min_trades=5):
    """
    Sweep entry thresholds and return the best in-sample boundary.
    The sweep is scored on the first `split_frac` of the sample only.

    Returns (best_threshold, sweep_frame, split_date).
    """
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}. Choose from {sorted(OBJECTIVES)}."
        )
    score_key = OBJECTIVES[objective]

    spread_frame = compute_spread(
        log_data, x_ticker, y_ticker, beta_lookback, signal_lookback
    )

    valid = spread_frame["z"].first_valid_index()
    live = spread_frame.loc[valid:]
    split_pos = min(max(int(len(live) * split_frac), 1), len(live) - 1)
    split_date = live.index[split_pos]

    if grid is None:
        grid = np.round(np.arange(1.0, 3.05, 0.1), 1)

    rows = []
    for threshold in grid:
        threshold = float(round(float(threshold), 1))
        if threshold >= stop_loss:
            continue

        positions, _ = run_state_machine(
            spread_frame["z"], threshold, exit_thresh, stop_loss, cooldown
        )
        _, net_returns, book = calculate_equity(
            log_data, spread_frame, positions, x_ticker, y_ticker,
            cost_bps=cost_bps, execution_lag=execution_lag,
        )

        in_sample = net_returns.loc[:split_date]
        in_book = book.loc[:split_date]
        metrics = calculate_tearsheet(in_sample, in_book, risk_free=risk_free)

        if metrics["Trades"] < min_trades:
            continue

        rows.append({
            "entry_threshold": threshold,
            "score": metrics[score_key],
            "Total Return": metrics["Total Return"],
            "Sortino Ratio": metrics["Sortino Ratio"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Trades": metrics["Trades"],
        })

    sweep = pd.DataFrame(rows)

    if sweep.empty:
        return 2.0, sweep, split_date

    best = float(sweep.loc[sweep["score"].idxmax(), "entry_threshold"])
    return round(best, 1), sweep, split_date
