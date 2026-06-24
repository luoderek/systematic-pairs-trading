import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from strategy_engine import generate_signals, calculate_equity, calculate_tearsheet, run_strategy_factory, test_cointegration


@st.cache_data(show_spinner=False)
def fetch_and_prep_data(ticker_x, ticker_y, start_date="2016-01-01"):
    """Fetches live market data and converts it to log prices."""
    raw_data = yf.download([ticker_x, ticker_y], start=start_date)['Close']
    clean_data = raw_data.dropna()
    log_data = np.log(clean_data)

    return log_data


# Page configuration & state management
st.set_page_config(page_title="StatArb Execution Engine", layout="wide")
st.title("Systematic Pairs Trading Optimiser")
st.markdown(
    "Automated cointegration mapping and risk-reward optimisation")
st.markdown("Tool created by Derek Luo")

if "dynamic_entry" not in st.session_state:
    st.session_state.dynamic_entry = 2.0

# Sidebar & data ingestion
st.sidebar.header("Asset Selection")
colA, colB = st.sidebar.columns(2)
with colA:
    x_ticker = st.text_input("Asset X \n(Independent)", value="ADI").upper()
with colB:
    y_ticker = st.text_input("Asset Y \n(Dependent)", value="AMD").upper()

# Dynamic data ingestion
with st.spinner(f"Fetching live market data for {x_ticker} and {y_ticker}..."):
    try:
        log_data = fetch_and_prep_data(x_ticker, y_ticker)
    except Exception as e:
        st.error(
            f"Failed to fetch data. Please check ticker symbols. Error: {e}")
        st.stop()


# Cointegration gatekeeper
st.sidebar.markdown("---")
st.sidebar.header("Structural Viability (1-Year Lookback)")

with st.spinner("Running Engle-Granger ADF test..."):
    # Isolate the last 252 days to prevent macro-drift from breaking the math
    recent_y = log_data[y_ticker].tail(252)
    recent_x = log_data[x_ticker].tail(252)

    coint_pvalue = test_cointegration(recent_y, recent_x)

if coint_pvalue < 0.05:
    st.sidebar.success(
        f"**Passed:** Stationary spread confirmed.\n\nP-Value: {coint_pvalue:.4f}")
elif coint_pvalue < 0.10:
    st.sidebar.warning(
        f"**Marginal:** Weak mean-reversion profile.\n\nP-Value: {coint_pvalue:.4f}")
else:
    st.sidebar.error(
        f"**Failed:** Non-stationary spread. Spurious correlation risk.\n\nP-Value: {coint_pvalue:.4f}")

st.sidebar.markdown("---")

# Strategy
st.sidebar.header("The Strategy Factory")
st.sidebar.markdown("Select an optimisation engine:")

# Engine 1: Maximum Profit
if st.sidebar.button("Run Max Profit"):
    with st.spinner("Optimising for Absolute Return..."):
        optimal_boundary = run_strategy_factory(
            log_data, x_ticker, y_ticker, objective='max profit')
        st.session_state.dynamic_entry = optimal_boundary
        st.sidebar.success(
            f"Max Profit Optimal Boundary: {optimal_boundary:.1f} SD")

# Engine 2: Maximum Sortino Ratio
if st.sidebar.button("Run Max Sortino"):
    with st.spinner("Optimising for Downside Protection..."):
        optimal_boundary = run_strategy_factory(
            log_data, x_ticker, y_ticker, objective='sortino')
        st.session_state.dynamic_entry = optimal_boundary
        st.sidebar.success(
            f"Sortino Optimal Boundary: {optimal_boundary:.1f} SD")

st.sidebar.markdown("---")
st.sidebar.header("Execution Thresholds")

entry_thresh = st.sidebar.slider(
    "Entry Boundary (Z-Score)",
    min_value=1.0,
    max_value=3.5,
    step=0.1,
    key="dynamic_entry"
)

exit_thresh = st.sidebar.slider(
    "Early Harvest Exit (Z-Score)", min_value=-0.5, max_value=1.5, value=0.0, step=0.1)
stop_loss = st.sidebar.slider(
    "Catastrophic Stop Loss", min_value=2.5, max_value=5.0, value=3.0, step=0.1)

# Backend execution
with st.spinner("Processing execution logic..."):
    optimal_boundary = st.session_state.get('dynamic_entry', 2.0)

    z_score, positions, exec_beta, _ = generate_signals(
        log_data, x_ticker, y_ticker, entry_thresh=optimal_boundary
    )

    equity_curve, daily_returns = calculate_equity(
        log_data, positions, exec_beta, x_ticker, y_ticker
    )

# Frontend visualisation
col1, col2 = st.columns(2)

with col1:
    st.subheader("Strategy Equity Curve (Net Returns)")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values * 100, mode='lines', line=dict(color='green')))
    fig_equity.update_layout(yaxis_title="Cumulative Return (%)",
                             template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_equity, use_container_width=True)

with col2:
    st.subheader("Dynamic Signal Oscillator (Z-Score)")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=z_score.index, y=z_score.values,
                    mode='lines', line=dict(color='purple', width=1), name="Z-Score"))
    fig_z.add_hline(y=entry_thresh, line_dash="dash",
                    line_color="red", annotation_text="Short Spread")
    fig_z.add_hline(y=-entry_thresh, line_dash="dash",
                    line_color="green", annotation_text="Long Spread")
    fig_z.add_hline(y=0, line_color="white", opacity=0.3)
    fig_z.update_layout(yaxis_title="Standard Deviations",
                        template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_z, use_container_width=True)

# Quantitative Tearsheet (KPI cards)
st.markdown("---")
st.subheader("Risk-Adjusted Performance Metrics")

tearsheet = calculate_tearsheet(daily_returns)
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi1.metric(label="Total Return",
            value=f"{tearsheet['Total Return'] * 100:.2f}%")
kpi2.metric(label="Max Drawdown",
            value=f"{tearsheet['Max Drawdown'] * 100:.2f}%")
kpi3.metric(label="Sharpe Ratio", value=f"{tearsheet['Sharpe Ratio']:.2f}")
kpi4.metric(label="Ann. Volatility",
            value=f"{tearsheet['Annualised Volatility'] * 100:.2f}%")
kpi5.metric(label="Win Rate", value=f"{tearsheet['Win Rate'] * 100:.1f}%")
kpi6.metric(label="Sortino", value=f"{tearsheet['Sortino Ratio']:.2f}")
