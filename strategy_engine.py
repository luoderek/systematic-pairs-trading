import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import statsmodels.tsa.stattools as ts


def test_cointegration(series_y, series_x):
    """
    Performs the Engle-Granger two-step cointegration test on log prices.
    Returns the p-value.
    """
    _, pvalue, _ = ts.coint(series_y, series_x)
    return pvalue


def generate_signals(log_data, x_ticker, y_ticker, beta_lookback=252, signal_lookback=30, entry_thresh=2.0, exit_thresh=0.0, stop_loss=3.0):
    """
    Computes rolling parameters, generates the dynamic spread, and runs the state machine.
    """
    # Vectorised Rolling OLS
    y_log = log_data[y_ticker]
    x_log = sm.add_constant(log_data[x_ticker])

    rolling_engine = RollingOLS(y_log, x_log, window=beta_lookback).fit()
    rolling_alpha = rolling_engine.params['const']
    rolling_beta = rolling_engine.params[x_ticker]

    # Dynamic Spread & Z-Score
    dynamic_spread = y_log - \
        (rolling_alpha + rolling_beta * log_data[x_ticker])
    spread_mean = dynamic_spread.rolling(window=signal_lookback).mean()
    spread_std = dynamic_spread.rolling(window=signal_lookback).std()
    z_score = (dynamic_spread - spread_mean) / spread_std

    # Path-Dependent State Machine
    z_array = z_score.to_numpy()
    positions = np.zeros(len(z_array))
    current_position = 0

    for i in range(len(z_array)):
        z = z_array[i]
        if np.isnan(z):
            positions[i] = 0
            continue

        # Flat state
        if current_position == 0:
            if entry_thresh <= z < stop_loss:
                # Spread is overpriced -> SHORT THE SPREAD
                current_position = -1
            elif -entry_thresh >= z > -stop_loss:
                # Spread is underpriced -> BUY THE SPREAD
                current_position = 1

        # Long state
        elif current_position == 1:
            # Take profit if Z rises above target, or stop if it crashes further
            if z >= -exit_thresh:
                current_position = 0
            elif z <= -stop_loss:
                current_position = 0

        # Short state
        elif current_position == -1:
            # Take profit if Z falls below target, or stop if it squeezes higher
            if z <= exit_thresh:
                current_position = 0
            elif z >= stop_loss:
                current_position = 0

        positions[i] = current_position

    return z_score, pd.Series(positions, index=log_data.index), rolling_beta, log_data


def calculate_equity(log_data, positions, execution_beta, x_ticker, y_ticker):
    """
    Shifts the position vector to prevent lookahead bias and calculates the compounding equity curve.
    """
    daily_returns_y = log_data[y_ticker].diff()
    daily_returns_x = log_data[x_ticker].diff()

    trade_positions = positions.shift(1)
    exec_beta = execution_beta.shift(1)

    spread_returns = daily_returns_y - (exec_beta * daily_returns_x)
    strategy_returns = trade_positions * spread_returns

    cumulative_return = np.exp(strategy_returns.cumsum()) - 1

    return cumulative_return, strategy_returns.fillna(0)


def calculate_tearsheet(strategy_returns):
    """
    Calculates core quantitative performance metrics from a series of daily returns.
    """
    cumulative_equity = np.exp(strategy_returns.cumsum())
    total_return = cumulative_equity.iloc[-1] - 1.0
    ann_volatility = strategy_returns.std() * np.sqrt(252)

    sharpe_ratio = (strategy_returns.mean() /
                    (strategy_returns.std() + 1e-9)) * np.sqrt(252)

    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 1:
        downside_vol = downside_returns.std() * np.sqrt(252)
    else:
        downside_vol = 0.0

    sortino_ratio = (strategy_returns.mean() * 252) / (downside_vol + 1e-9)

    rolling_max = cumulative_equity.cummax()
    max_drawdown = ((cumulative_equity - rolling_max) / rolling_max).min()

    winning_days = len(strategy_returns[strategy_returns > 0])
    active_days = len(strategy_returns[strategy_returns != 0])
    win_rate = winning_days / active_days if active_days > 0 else 0

    return {
        "Total Return": total_return,
        "Annualised Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def run_strategy_factory(log_data, x_ticker, y_ticker, objective='sortino'):
    """
    Goes through Z-Score thresholds to find the optimal entry boundary.
    """
    best_score = -float('inf')
    best_threshold = 2.0

    for threshold in np.arange(1.0, 3.1, 0.1):
        _, positions, execution_beta, log_data = generate_signals(
            log_data, x_ticker, y_ticker, entry_thresh=threshold
        )

        _, daily_returns = calculate_equity(
            log_data, positions, execution_beta, x_ticker, y_ticker
        )

        metrics = calculate_tearsheet(daily_returns)

        if metrics['total_trades'] == 0:
            continue

        if objective == 'max profit':
            current_score = metrics['Total Return']
        elif objective == 'sortino':
            current_score = metrics['Sortino Ratio']

        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold

    return best_threshold
