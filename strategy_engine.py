import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import statsmodels.tsa.stattools as ts

TRADING_DAYS = 252
EPS = 1e-12


# ---- Structural Tests ----

def test_cointegration(series_y, series_x):
    """
    Engle-Granger test on log prices. Returns the p-value.

    A small p-value means the spread is stationary, so it should pull back
    towards its mean instead of wandering off. Returns NaN when there isn't
    enough clean data for the test to mean anything.
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
    Roughly how many days the spread takes to close half the gap back to zero.

    Regress today's change on yesterday's level. The slope (kappa) comes out
    negative when the spread is being pulled back. If it's positive there is
    no mean reversion at all, so return inf instead of a made up number.

    Useful for picking signal_lookback. A 30 day z-score window on a spread
    with a 90 day half-life is going to be normalising against the wrong mean
    most of the time.
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


# ---- Spread Construction ----

def compute_spread(log_data, x_ticker, y_ticker, beta_lookback=252, signal_lookback=30):
    """
    Rolling hedge ratio, spread, and z-score.

    None of this depends on the entry threshold, so the factory can build it
    once and reuse it for every threshold it tests instead of refitting the
    regression 21 times.

    Returns a DataFrame with alpha, beta, spread, z.
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

    # TODO: regressing Y on X and X on Y gives two different betas, so neither
    # is really the hedge ratio. Read up on Johansen / total least squares.
    fit = RollingOLS(y_log, x_log, window=beta_lookback).fit()
    alpha = fit.params["const"]
    beta = fit.params[x_ticker]

    spread = y_log - (alpha + beta * frame[x_ticker])
    mean = spread.rolling(window=signal_lookback).mean()
    std = spread.rolling(window=signal_lookback).std()

    # If the spread is flat across a whole window the std is 0 and the z-score
    # blows up to inf. Blank it instead.
    std = std.where(std > EPS)
    z = (spread - mean) / std

    return pd.DataFrame(
        {"alpha": alpha, "beta": beta, "spread": spread, "z": z},
        index=frame.index,
    )


# ---- State Machine ----

def run_state_machine(z, entry_thresh, exit_thresh=0.0, stop_loss=3.0, cooldown=0):
    """
    Turns the z-score into positions: -1 short spread, 0 flat, +1 long spread.

    To open, z has to be past entry_thresh but still inside stop_loss. If the
    spread has already gapped past the stop we skip it. No point buying
    into something that's clearly broken.

    cooldown blocks new entries for N days after a stop, so it doesn't keep
    jumping straight back into the same losing spread.

    Returns positions and exit_reasons, where exit_reasons is 'stop' or
    'target' on the day the position closes.
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
            # check the stop first, otherwise a day that blows through both
            # gets logged as a winner
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
    """Spread and state machine in one call. Handy for quick tests."""
    spread_frame = compute_spread(
        log_data, x_ticker, y_ticker, beta_lookback, signal_lookback
    )
    positions, reasons = run_state_machine(
        spread_frame["z"], entry_thresh, exit_thresh, stop_loss, cooldown
    )
    return spread_frame, positions, reasons


# ---- Portfolio and P&L ----

def calculate_equity(log_data, spread_frame, positions, x_ticker, y_ticker,
                     cost_bps=5.0, execution_lag=1):
    """
    Turn positions into a daily net return series.

    One unit of position means 1 of Y against |beta| of X, so the money tied
    up is (1 + |beta|), not 1. Dividing by that gives return on the capital
    actually used. Skip it and the numbers look better than they are any time
    beta is above 1.

    execution_lag=1 means the signal from the close of day T is filled at that
    same close and earns the move into T+1. Nothing looks forward, but it does
    assume you can trade the closing print. Set it to 2 to wait a full day.

    cost_bps is commission plus half the bid-ask, per leg.

    Returns equity_curve, net_returns, and book (the daily weights and costs).
    """
    if execution_lag < 1:
        raise ValueError(
            "execution_lag must be at least 1 to avoid lookahead bias.")

    index = spread_frame.index
    y_log = log_data[y_ticker].reindex(index)
    x_log = log_data[x_ticker].reindex(index)

    # Convert to simple returns first. Log returns can't be added across two
    # legs, only across time.
    ret_y = np.exp(y_log.diff()) - 1.0
    ret_x = np.exp(x_log.diff()) - 1.0

    held = positions.shift(execution_lag)
    beta_exec = spread_frame["beta"].shift(execution_lag)

    gross = 1.0 + beta_exec.abs()
    weight_y = (held / gross).fillna(0.0)
    weight_x = (-held * beta_exec / gross).fillna(0.0)

    gross_return = weight_y * ret_y + weight_x * ret_x

    # Costs come off the change in both weights. Beta moves a bit every day,
    # so there's a small charge even when the position is unchanged.
    turnover = (weight_y.diff().abs() + weight_x.diff().abs()).fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)

    net_returns = (gross_return - costs).fillna(0.0)

    # Cut off the warm-up. Those days are forced flat while the rolling
    # windows fill up, and leaving a year of zeros in pulls the vol down.
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


# ---- Performance Metrics ----

def downside_deviation(returns, target=0.0, periods=TRADING_DAYS):
    """
    Downside deviation, annualised.

    Root mean square of everything below target, divided by the total number
    of days including the good ones.

    Dividing by only the losing days is what broke my Sortino. That version
    measures how spread out the losses are rather than how often they happen,
    and it barely moves when you change the entry threshold, so Sortino ends
    up as total return with a different scale and both optimisers pick the
    same answer.
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
    Performance stats from a series of daily net returns.

    risk_free is annual and decimal, so 0.04 for 4%. It comes off before
    Sharpe and Sortino so both are excess return ratios.

    Pass `book` as well to get the trade-level numbers. Win rate without it is
    per day, which is not the same thing and reads much higher.
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

    # sample std (ddof=1). Barely changes anything at this sample size but
    # it's the right one for an estimate.
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
    Group the daily book into round-trip trades.

    A trade is just a run of days sitting at the same non-zero position.
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

        # The unwind trades on the first flat day, so grab that day's cost as
        # well. Without this the trades don't add back up to the equity curve
        # and the win rate comes out too high.
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


# ----Tthreshold Optimisation ----

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
    Try every entry threshold on the grid and return whichever scores best.

    Only the first split_frac of the data gets scored. The rest is held back
    so the app can show a result the optimiser never looked at. Picking the
    threshold on the full history and then reporting on that same history
    would just be reporting the best fit.

    Returns best_threshold, the sweep table, and the split date.
    """
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective {objective!r}. Choose from {sorted(OBJECTIVES)}."
        )
    score_key = OBJECTIVES[objective]

    # built once and reused, it doesn't depend on the threshold
    spread_frame = compute_spread(
        log_data, x_ticker, y_ticker, beta_lookback, signal_lookback
    )

    valid = spread_frame["z"].first_valid_index()
    live = spread_frame.loc[valid:]
    split_pos = min(max(int(len(live) * split_frac), 1), len(live) - 1)
    split_date = live.index[split_pos]

    if grid is None:
        grid = np.round(np.arange(1.0, 3.05, 0.1), 1)

    # TODO: 21 tries on the same data means the best in-sample score is partly
    # luck. The held-out window helps but doesn't fix it. Look at deflated
    # Sharpe for a proper correction.
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

        # a threshold that only fires twice can post a great score by luck
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
