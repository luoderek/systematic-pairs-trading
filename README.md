# Systematic Pairs Trading & Optimisation Engine

## Overview
This repository contains a path-dependent systematic trading engine and interactive dashboard designed to identify, hedge, and extract value from mean-reverting anomalies in non-stationary time series. By utilising the Engle-Granger two-step method, the pipeline isolates stationary residual spreads between asset pairs and executes dynamic, market-neutral hedging through rolling regression. 

The core system is abstracted into an interactive Streamlit application, enabling rapid parameter sweeps, dynamic equity curve rendering, and dual-metric risk optimisation.

## Mathematical Framework

**Cointegration & Dynamic Hedging**
The engine does not rely on simple asset correlation; it hunts for structural stationarity. It utilises the Engle-Granger test to confirm that the spread between two historically non-stationary time series is mean-reverting. To maintain strict market neutrality across shifting macroeconomic regimes, the system calculates a dynamic hedge ratio ($\beta$) using a rolling Ordinary Least Squares (OLS) regression.

The spread is calculated dynamically at time $t$ as:
$$Spread_t = \ln(Y_t) - \beta_t \ln(X_t)$$

**Signal Generation**
The raw spread is normalised into a rolling Z-score oscillator. This allows the state machine to trigger execution logic universally across any asset pair without hardcoding arbitrary price boundaries. 

$$Z_t = \frac{Spread_t - \mu_{rolling}}{\sigma_{rolling}}$$

## System Architecture & Execution Logic

**Path-Dependent State Machine**
The backend relies on a strictly path-dependent execution state machine to dictate portfolio positioning (Flat, Long Spread, Short Spread). It mathematically prevents conflicting signals and ensures the portfolio accurately reflects real-world trading logic.

**Lookahead Bias Mitigation**
To ensure backtesting integrity, the system architecture explicitly addresses lookahead bias. The logic arrays are engineered with a strict $T+1$ execution shift. Signals generated at the close of day $T$ are executed at the closing price of day $T+1$, simulating real-world execution friction and preventing the algorithm from accessing future data points.

## Risk Optimisation (Max Profit vs. Sortino)

The engine features a dual-metric optimisation framework that sweeps standard deviation entry thresholds (e.g., 1.0 to 3.0 SD) to evaluate system behavior under different risk profiles. 

1. **Maximum Profit Objective:** Optimises strictly for the highest cumulative return, serving as a baseline performance metric.
2. **Sortino Ratio Optimisation:** Rejects the industry-standard Sharpe ratio to actively target and penalise downside volatility. This secondary engine acts as an automated risk desk, sacrificing marginal basis points of raw profit to secure a mathematically smoother equity curve and protect against catastrophic tail-risk.

## Technical Stack
* **Language:** Python
* **Data Processing & Array Mathematics:** Pandas, NumPy
* **Statistical Modelling:** Statsmodels (OLS, Cointegration)
* **Web UI & Visualisation:** Streamlit, Plotly

## Local Usage

1. Clone the repository:
```bash
   git clone https://github.com/luoderek/systematic-pairs-trading.git
```
2. Navigate to the project directory and install the required dependencies:
```bash
   pip install -r requirements.txt
```
3. Initialise the Streamlit application:
```bash
   streamlit run app.py
```
