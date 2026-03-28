import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetcher import fetch_market_data, fetch_asset_info
from data.processor import generate_export_report, calculate_portfolio_dividend, validate_tickers
from models.markowitz import optimize_markowitz, optimize_markowitz_constrained
from models.black_litterman import optimize_black_litterman
from models.hrp import optimize_hrp
from models.risk_parity import optimzie_risk_parity
from plots.standard_plots import plot_monte_carlo_ef, plot_backtest

# --- Page Config ---
st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")
st.title("Portfolio Optimization Dashboard")

# --- Hardcoded Top US Tickers (Top 100 proxy by market cap) ---
TOP_100_US = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK-B', 'LLY', 'AVGO', 'JPM', 
    'TSLA', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD', 'COST', 'MRK', 'ABBV', 'CRM', 
    'CVX', 'AMD', 'NFLX', 'PEP', 'KO', 'BAC', 'WMT', 'TMO', 'MCD', 'CSCO', 'ACN', 
    'LIN', 'INTC', 'ABT', 'ORCL', 'CMCSA', 'DIS', 'TXN', 'DHR', 'PFE', 'AMGN', 'VZ', 
    'NKE', 'IBM', 'NEE', 'PM', 'UNP', 'HON', 'SPGI', 'RTX', 'QCOM', 'INTU', 'CAT', 
    'GE', 'NOW', 'AMAT', 'UBER', 'GS', 'BA', 'MS', 'BKNG', 'T', 'AXP', 'ISRG', 'SYK', 
    'BLK', 'MDLZ', 'TJX', 'MMC', 'C', 'VRTX', 'REGN', 'LMT', 'ADI', 'PGR', 'CVS', 
    'ZTS', 'BSX', 'CB', 'CI', 'FI', 'ETN', 'SLB', 'GILD', 'BDX', 'DE', 'MU', 'SO'
]

# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

selected_list = st.sidebar.multiselect(
    "Select from Top 100 US Stocks",
    TOP_100_US,
    default=['AAPL', 'MSFT']
)

custom_tickers = st.sidebar.text_input("Add Custom Tickers (comma separated)", "SPY")

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

model_choice = st.sidebar.selectbox(
    "Optimization Model",
    ["Markowitz (Max Sharpe)", "Hierarchical Risk Parity", "Risk Parity", "Black-Litterman"]
)

apply_constraints = st.sidebar.checkbox("Apply Sector Constraints (Max 30%)", value=False)

# --- Execution ---
if st.sidebar.button("Run Optimization"):
    # 1. Smash the lists together and remove duplicates
    raw_custom = [t.strip().upper() for t in custom_tickers.split(',')] if custom_tickers else []
    all_tickers = list(set(selected_list + raw_custom))
    all_tickers = [t for t in all_tickers if t]

    # 2. Check if a ticker exists
    with st.spinner("Checking if your custom tickers actually exist..."):
        valid_tickers, invalid_tickers = validate_tickers(all_tickers)

    if invalid_tickers:
        # Shame the user
        st.sidebar.warning(f"Ignored invalid or delisted tickers: {', '.join(invalid_tickers)}")
        
    if len(valid_tickers) < 2:
        st.sidebar.error("You need at least 2 valid tickers to run an optimization. Try again.")
        st.stop()

    with st.spinner("Running matrix algebra. Please hold..."):
        prices, returns = fetch_market_data(valid_tickers, start_date, end_date)
        spy_prices, _ = fetch_market_data(['SPY'], start_date, end_date)
        sector_map, div_yields = fetch_asset_info(valid_tickers)

        # 2. Route to Model
        if model_choice == "Markowitz (Max Sharpe)":
            if apply_constraints:
                weights, performance = optimize_markowitz_constrained(prices, sector_map)
            else:
                weights, performance = optimize_markowitz(prices)

        elif model_choice == "Hierarchical Risk Parity":
            weights, performance = optimize_hrp(prices, returns)

        elif model_choice == "Risk Parity":
            weights, performance = optimzie_risk_parity(prices)

        elif model_choice == "Black-Litterman":
            st.info("Using baseline market caps and neutral 5% views for demo stability.")
            mcaps = {t: 1000000000 for t in valid_tickers}
            views = {t: 0.05 for t in valid_tickers}

            weights, performance = optimize_black_litterman(prices, spy_prices, mcaps, views)

        # 3. KPI Cards
        port_div_yield = calculate_portfolio_dividend(weights, div_yields)
        expected_return, volatility, sharpe = performance
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
        col2.metric("Annual Volatility", f"{volatility*100:.2f}%")
        col3.metric("Sharpe Ration", f"{sharpe:.2f}")
        col4.metric("Dividend Yield", f"{port_div_yield*100:.2f}%")

        # 4. Visuals
        st.markdown("---")
        chart_col, weight_col = st.columns([2, 1])

        with chart_col:
            st.subheader("Asset Correlation Matrix")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.subheader("Efficient Frontier")
            ef_fig = plot_monte_carlo_ef(prices)
            st.pyplot(ef_fig)

        with weight_col:
            st.subheader("Optimal Weights")
            st.dataframe(weights, width=True)

            csv_data = generate_export_report(weights, performance, 0.0)
            st.download_button(
                label="Download Report (CSV)",
                data=csv_data,
                file_name="optimized_portfolio.csv",
                mime="text/csv"
            )

        # 5. Backtest    
        st.markdown("---")
        st.subheader("Historical Performance Simulation")
        backtest_fig = plot_backtest(returns, weights, spy_prices)
        st.pyplot(backtest_fig)