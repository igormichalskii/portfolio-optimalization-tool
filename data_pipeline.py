import yfinance as yf
import pandas as pd

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

if __name__ == "__main__":
    # Example of usage
    prices, returns = fetch_market_data(['SPY', 'AAPL', 'TSLA', 'QQQ'], '2020-01-01', '2024-01-01')
    print(f"Printing prices: \n{prices}.\nPrinting returns: \n{returns}")