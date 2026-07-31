import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import strategy_engine as se

st.set_page_config(page_title="StatArb Execution Engine", layout="wide")


# ---- Data ----

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_log_prices(ticker_x, ticker_y, start_date, end_date):
    """
    Pull adjusted closes and take logs.

    auto_adjust is set here on purpose. yfinance has changed the default
    between versions, and unadjusted prices break the cointegration test
    because dividends leave steps in the series.
    """
    raw = yf.download(
        [ticker_x, ticker_y],
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            "No data returned. Check the ticker symbols and date range.")

    close = raw["Close"] if isinstance(
        raw.columns, pd.MultiIndex) else raw[["Close"]]
    missing = [t for t in (ticker_x, ticker_y) if t not in close.columns]
    if missing:
        raise ValueError(f"No price history found for: {', '.join(missing)}")

    clean = close[[ticker_x, ticker_y]].dropna()
    if (clean <= 0).any().any():
        raise ValueError("Non-positive prices encountered; cannot take logs.")

    return np.log(clean)


@st.cache_data(show_spinner=False)
def cached_spread(log_data, x, y, beta_lookback, signal_lookback):
    return se.compute_spread(log_data, x, y, beta_lookback, signal_lookback)


@st.cache_data(show_spinner=False)
def cached_factory(log_data, x, y, objective, beta_lookback, signal_lookback,
                   exit_thresh, stop_loss, cooldown, cost_bps, execution_lag,
                   risk_free, split_frac):
    return se.run_strategy_factory(
        log_data, x, y, objective=objective, beta_lookback=beta_lookback,
        signal_lookback=signal_lookback, exit_thresh=exit_thresh,
        stop_loss=stop_loss, cooldown=cooldown, cost_bps=cost_bps,
        execution_lag=execution_lag, risk_free=risk_free, split_frac=split_frac,
    )


# ---- Header ----

st.title("Systematic Pairs Trading Optimiser")
st.caption(
    "Cointegration screening, dynamic hedging, and threshold optimisation with "
    "out-of-sample validation. Tool created by Derek Luo."
)

DEFAULTS = {
    "entry_thresh": 2.0,
    "exit_thresh": 0.0,
    "stop_loss": 3.0,
    "cooldown": 0,
    "cost_bps": 5.0,
    "execution_lag": 1,
    "risk_free": 0.0,
    "split_frac": 70,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("last_optimisation", None)
st.session_state.setdefault("sweep", None)


# ---- Sidebar: Assets ----

st.sidebar.header("Asset Selection")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    x_ticker = st.text_input("Asset X (independent)",
                             value="ADI").strip().upper()
with col_b:
    y_ticker = st.text_input("Asset Y (dependent)",
                             value="AMD").strip().upper()

date_col1, date_col2 = st.sidebar.columns(2)
with date_col1:
    start_date = st.date_input("Start", value=pd.Timestamp("2016-01-01"))
with date_col2:
    end_date = st.date_input("End", value=pd.Timestamp.today())

if not x_ticker or not y_ticker:
    st.info("Enter two ticker symbols to begin.")
    st.stop()

# Same ticker twice gives beta = 1 and a spread of zero, so the z-score is 0/0
if x_ticker == y_ticker:
    st.error("Asset X and Asset Y must be different tickers.")
    st.stop()

with st.spinner(f"Fetching {x_ticker} and {y_ticker}..."):
    try:
        log_data = fetch_log_prices(x_ticker, y_ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        st.stop()


# ---- Sidebar: Windows and Costs ----

st.sidebar.markdown("---")
st.sidebar.header("Model Windows")
beta_lookback = st.sidebar.number_input(
    "Hedge ratio window (days)", min_value=60, max_value=756, value=252, step=21,
    help="How much history the rolling regression uses to set the hedge ratio.",
)
signal_lookback = st.sidebar.number_input(
    "Z-score window (days)", min_value=10, max_value=252, value=30, step=5,
    help="Days used for the mean and standard deviation of the spread. Works "
         "best somewhere near the half-life shown below.",
)

try:
    spread_frame = cached_spread(
        log_data, x_ticker, y_ticker, int(beta_lookback), int(signal_lookback)
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Execution Frictions")
st.sidebar.slider(
    "Round-trip cost per leg (bps)", min_value=0.0, max_value=50.0, step=0.5,
    key="cost_bps",
    help="Commission plus half the bid-ask spread. Charged on both legs, "
         "including the small daily rebalance as the hedge ratio drifts.",
)
st.sidebar.selectbox(
    "Execution lag", options=[1, 2], key="execution_lag",
    format_func=lambda v: "T+0 fill (signal close)" if v == 1 else "T+1 fill (next close)",
    help="T+0 assumes you can trade the same close that gave you the signal. "
         "T+1 waits a full day before filling.",
)
st.sidebar.number_input(
    "Risk-free rate (annual, %)", min_value=0.0, max_value=10.0, step=0.25,
    key="risk_free",
    help="Taken off the returns before Sharpe and Sortino are calculated.",
)
st.sidebar.slider(
    "In-sample fraction (%)", min_value=40, max_value=90, step=5, key="split_frac",
    help="The optimiser only sees this much of the data. The rest is kept back.",
)


# ---- Sidebar: Is the pair worth trading ----

st.sidebar.markdown("---")
st.sidebar.header("Structural Viability")

split_frac = st.session_state.split_frac / 100.0
first_valid = spread_frame["z"].first_valid_index()
live_index = spread_frame.loc[first_valid:].index
split_pos = min(max(int(len(live_index) * split_frac), 1), len(live_index) - 1)
split_date = live_index[split_pos]

# test the in-sample window separately from the last year. If the first one
# passes and the second doesn't, the relationship has broken down since.
formation_p = se.test_cointegration(
    log_data[y_ticker].loc[:split_date], log_data[x_ticker].loc[:split_date]
)
recent_p = se.test_cointegration(
    log_data[y_ticker].tail(252), log_data[x_ticker].tail(252)
)
hl = se.half_life(spread_frame["spread"].loc[:split_date])


def viability_box(label, pvalue):
    if np.isnan(pvalue):
        st.sidebar.warning(f"{label}: test could not be computed.")
    elif pvalue < 0.05:
        st.sidebar.success(f"{label}: stationary spread (p = {pvalue:.4f})")
    elif pvalue < 0.10:
        st.sidebar.warning(f"{label}: marginal (p = {pvalue:.4f})")
    else:
        st.sidebar.error(
            f"{label}: non-stationary, spurious risk (p = {pvalue:.4f})")


viability_box("In-sample window", formation_p)
viability_box("Trailing 1 year", recent_p)
st.sidebar.caption(
    f"Spread half-life: {hl:.1f} days."
    if np.isfinite(hl) else "Spread half-life: no mean reversion detected."
)
if np.isfinite(hl) and not (0.5 * hl <= signal_lookback <= 4 * hl):
    st.sidebar.caption(
        "Z-score window is far from the half-life; consider moving it closer."
    )


# ---- Sidebar: Optimisation ----

def optimise(objective):
    """
    Runs when one of the factory buttons is clicked.

    This has to be a callback. Setting entry_thresh further down the script
    throws an error once the slider that owns the key already exists.
    """
    stop = st.session_state.stop_loss
    best, sweep, _ = cached_factory(
        log_data, x_ticker, y_ticker, objective,
        int(beta_lookback), int(signal_lookback),
        st.session_state.exit_thresh, stop, int(st.session_state.cooldown),
        st.session_state.cost_bps, int(st.session_state.execution_lag),
        st.session_state.risk_free / 100.0, split_frac,
    )
    st.session_state.entry_thresh = float(np.clip(best, 1.0, stop - 0.1))
    st.session_state.sweep = sweep
    st.session_state.last_optimisation = (objective, best, len(sweep))


st.sidebar.markdown("---")
st.sidebar.header("Strategy Factory")
st.sidebar.caption("Thresholds are fitted on the in-sample window only.")

st.sidebar.button(
    "Optimise: Max Return", width='stretch',
    on_click=optimise, args=("max profit",),
)
st.sidebar.button(
    "Optimise: Max Sortino", width='stretch',
    on_click=optimise, args=("sortino",),
)
st.sidebar.button(
    "Optimise: Max Sharpe", width='stretch',
    on_click=optimise, args=("sharpe",),
)

if st.session_state.last_optimisation:
    objective, best, tested = st.session_state.last_optimisation
    if tested == 0:
        st.sidebar.error(
            "No threshold produced enough trades in-sample. Widen the date "
            "range or loosen the stop."
        )
    else:
        st.sidebar.success(
            f"{objective.title()} optimum: {best:.1f} SD ({tested} thresholds tested)"
        )


# ---- Sidebar: Thresholds ----

st.sidebar.markdown("---")
st.sidebar.header("Execution Thresholds")

st.sidebar.slider(
    "Catastrophic stop loss (Z)", min_value=2.0, max_value=6.0, step=0.1,
    key="stop_loss",
)

# entry has to sit inside the stop or the entry condition can never be true.
# clamp it here, before the slider is built, because you can't change a
# widget's value after it exists.
entry_ceiling = round(st.session_state.stop_loss - 0.1, 1)
st.session_state.entry_thresh = float(
    np.clip(st.session_state.entry_thresh, 1.0, entry_ceiling)
)

st.sidebar.slider(
    "Entry boundary (Z)", min_value=1.0, max_value=entry_ceiling, step=0.1,
    key="entry_thresh",
    help="Capped just under the stop. An entry at or past the stop can never "
         "fire, and you'd get a flat equity curve with no warning.",
)
st.sidebar.slider(
    "Early harvest exit (Z)", min_value=-0.5, max_value=1.5, step=0.1,
    key="exit_thresh",
    help="Above zero closes the trade before the spread gets all the way back "
         "to its mean.",
)
st.sidebar.number_input(
    "Cooldown after stop (days)", min_value=0, max_value=60, step=1, key="cooldown",
    help="Waits this many days before allowing a new trade after a stop.",
)


# ---- Run the Backtest ----

with st.spinner("Running backtest..."):
    positions, exit_reasons = se.run_state_machine(
        spread_frame["z"],
        st.session_state.entry_thresh,
        st.session_state.exit_thresh,
        st.session_state.stop_loss,
        int(st.session_state.cooldown),
    )
    equity_curve, net_returns, book = se.calculate_equity(
        log_data, spread_frame, positions, x_ticker, y_ticker,
        cost_bps=st.session_state.cost_bps,
        execution_lag=int(st.session_state.execution_lag),
    )

risk_free = st.session_state.risk_free / 100.0
full = se.calculate_tearsheet(net_returns, book, risk_free=risk_free)
in_sample = se.calculate_tearsheet(
    net_returns.loc[:split_date], book.loc[:split_date], risk_free=risk_free
)
out_sample = se.calculate_tearsheet(
    net_returns.loc[split_date:], book.loc[split_date:], risk_free=risk_free
)

if full["Trades"] == 0:
    st.warning(
        "No trades were generated. The entry boundary may be too wide for this "
        "spread, or the sample too short after the warm-up window."
    )


# ---- Charts ----

left, right = st.columns(2)

with left:
    st.subheader("Equity Curve (net of costs)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values * 100,
        mode="lines", line=dict(color="#2ecc71", width=2), name="Strategy",
    ))
    # shade the held-out part so it's obvious which half is the honest one
    fig.add_vrect(
        x0=split_date, x1=equity_curve.index[-1],
        fillcolor="#ffffff", opacity=0.06, line_width=0,
        annotation_text="out-of-sample", annotation_position="top left",
    )
    fig.add_vline(x=split_date, line_dash="dot", line_color="#f1c40f")
    fig.update_layout(
        yaxis_title="Cumulative return (%)", template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Signal Oscillator (Z-Score)")
    entry = st.session_state.entry_thresh
    stop = st.session_state.stop_loss
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=spread_frame.index, y=spread_frame["z"].values,
        mode="lines", line=dict(color="#9b59b6", width=1), name="Z-score",
    ))
    for level, colour, label in [
        (entry, "#e74c3c", "Short spread"), (-entry, "#2ecc71", "Long spread"),
        (stop, "#e67e22", "Stop"), (-stop, "#e67e22", "Stop"),
    ]:
        fig_z.add_hline(y=level, line_dash="dash", line_color=colour,
                        annotation_text=label, annotation_position="right")
    fig_z.add_hline(y=0, line_color="white", opacity=0.3)
    fig_z.update_layout(
        yaxis_title="Standard deviations", template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
    )
    st.plotly_chart(fig_z, width='stretch')


# ---- Metrics Table ----

st.markdown("---")
st.subheader("Risk-Adjusted Performance")
st.caption(
    f"In-sample through {split_date.date()} "
    f"({in_sample['Observations']} days), held out afterwards "
    f"({out_sample['Observations']} days). Returns are net of "
    f"{st.session_state.cost_bps:.1f} bps per leg and divided by gross "
    f"exposure, so they're returns on the capital actually used."
)

summary = pd.DataFrame(
    {"In-sample": in_sample, "Out-of-sample": out_sample, "Full period": full}
).T

display = pd.DataFrame({
    "Total Return": summary["Total Return"].map("{:.2%}".format),
    "CAGR": summary["CAGR"].map("{:.2%}".format),
    "Max Drawdown": summary["Max Drawdown"].map("{:.2%}".format),
    "Ann. Vol": summary["Annualised Volatility"].map("{:.2%}".format),
    "Sharpe": summary["Sharpe Ratio"].map("{:.2f}".format),
    "Sortino": summary["Sortino Ratio"].map("{:.2f}".format),
    "Calmar": summary["Calmar Ratio"].map("{:.2f}".format),
    "Trades": summary["Trades"].astype(int),
    "Trade Win Rate": summary["Trade Win Rate"].map("{:.1%}".format),
    "Time in Market": summary["Time In Market"].map("{:.1%}".format),
    "Cost Drag": summary["Total Cost"].map("{:.2%}".format),
})
st.dataframe(display, width='stretch')

if out_sample["Trades"] >= 5 and in_sample["Sharpe Ratio"] > 0:
    decay = 1 - out_sample["Sharpe Ratio"] / in_sample["Sharpe Ratio"]
    if decay > 0.5:
        st.warning(
            f"Out-of-sample Sharpe is {decay:.0%} lower than in-sample. That "
            f"much of a drop usually means the threshold is overfitted."
        )


# ---- Detail Tabs ----

tab_sweep, tab_trades, tab_hedge = st.tabs(
    ["Threshold sweep", "Trade log", "Hedge ratio"]
)

with tab_sweep:
    sweep = st.session_state.sweep
    if sweep is None or sweep.empty:
        st.info("Run an optimiser to see how each objective scores the grid.")
    else:
        # two y-axes because return and Sortino are on completely different
        # scales. The point is where the peaks sit, not the levels.
        fig_s = go.Figure()
        for column, axis in [("Total Return", "y"), ("Sortino Ratio", "y2")]:
            fig_s.add_trace(go.Scatter(
                x=sweep["entry_threshold"], y=sweep[column],
                mode="lines+markers", name=column, yaxis=axis,
            ))
        fig_s.update_layout(
            template="plotly_dark", xaxis_title="Entry boundary (SD)",
            yaxis=dict(title="Total return"),
            yaxis2=dict(title="Sortino", overlaying="y", side="right"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_s, width='stretch')
        st.dataframe(sweep, width='stretch', hide_index=True)

with tab_trades:
    trades = se.summarise_trades(book)
    if trades.empty:
        st.info("No completed trades in this configuration.")
    else:
        stops = int((exit_reasons == "stop").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Completed trades", len(trades))
        c2.metric("Stopped out", stops)
        c3.metric("Median holding period",
                  f"{trades['days'].median():.0f} days")
        st.dataframe(
            trades.assign(
                **{"return": trades["return"].map("{:.2%}".format),
                   "cost": trades["cost"].map("{:.3%}".format)}
            ),
            width='stretch', hide_index=True,
        )

with tab_hedge:
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=spread_frame.index, y=spread_frame["beta"],
        mode="lines", line=dict(color="#3498db"), name="Rolling beta",
    ))
    fig_b.update_layout(
        template="plotly_dark", yaxis_title=f"Units of {x_ticker} per unit of {y_ticker}",
        margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
    )
    st.plotly_chart(fig_b, width='stretch')
    st.caption(
        "If beta jumps around a lot, the link between the two stocks isn't "
        "stable and the p-value on its own is hiding that."
    )

# TODO: no borrow cost on the short leg yet. Fine for large caps, would
# matter a lot for anything hard to borrow.
