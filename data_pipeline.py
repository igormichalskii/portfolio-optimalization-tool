import yfinance as yf
import pandas as pd
from pypfopt import expected_returns, risk_models, black_litterman, HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier, EfficientRiskParity

def fetch_market_data(tickers, start_date, end_date):
    # Download the data. We only care about Close, since now Close column is adjusted for splits and dividends.
    print(f"Fetching data for: {tickers}.")
    data = yf.download(tickers, start_date, end_date)['Close']

    # If the user only passed one ticker, yfinance returns a Series. Force it to DataFrame.
    if isinstance(data, pd.Series):
        data = data.toframe(tickers[0])

    # Scrub the data: forward-fill missing days (weekends/holidays/glitches), then drop whatever is still broken
    prices = data.ffill().dropna()
    # Calculate daily percentage returns
    returns = prices.pct_change().dropna()

    return prices, returns

def optimize_markowitz(prices):
    # Expected Returns: We assume that the past predicts the future.
    mu = expected_returns.mean_historical_return(prices)

    # Risk Model: Ledoit-Wolf, a mathematical trick to smooth out extreme outliers so the optimizer doesn't hallucinate.
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # The Optimizer
    ef = EfficientFrontier(mu, S)

    # Clean the weights: Round off microscopic fractions. Zeros out anything under 1%.
    cleaned_weights = ef.clean_weights()

    # Extract the metrics: Expected Annual Return, Annual Volatility, Sharpe Ratio
    performance = ef.portfolio_performance(verbose=False)

    return cleaned_weights, performance

def optimize_black_litterman(prices, benchmark_prices, mcaps, views):
    # Basic Covariance
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Market Implied Risk Aversion
    # Calculates how much risk the overall market currently tolerates.
    delta = black_litterman.market_implied_risk_aversion(benchmark_prices)

    # What the market expects these stocks to return based on their size and risk
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    # Mixes the prior with absolute views
    bl = BlackLittermanModel(S, pi=prior, absolute_views=views)
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Optimize using the new BL returns, and covariance
    ef = EfficientFrontier(bl_returns, bl_cov)
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)

    return cleaned_weights, performance

if __name__ == "__main__":
    # Example of usage
    prices, returns = fetch_market_data(['SPY', 'AAPL', 'TSLA', 'QQQ'], '2020-01-01', '2024-01-01')
    print(f"Printing prices: \n{prices}.\nPrinting returns: \n{returns}")