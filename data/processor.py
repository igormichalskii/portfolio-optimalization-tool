import pandas as pd
import yfinance as yf

def validate_tickers(tickers):
    valid, invalid = [], []
    for t in tickers:
        if not t:
            continue

        try:
            if not yf.Ticker(t).history(period="1d").empty:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception as e:
            invalid.append(t)
    
    return valid, invalid

def calculate_portfolio_dividend(weights, div_yields):
    """Calculates the weighted average dividend yield."""
    return sum(weights.get(t, 0) * div_yields.get(t, 0) for t in weights)

def generate_export_report(weights, performance, total_div_yield):
    """Packages the weights and matrics into a CSV-ready string."""
    df_weights = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Allocation'])
    df_weights['Allocation'] = df_weights['Allocation'].apply(lambda x: f"{x*100:.2f}%")

    metrics = [
        {"Ticker": "Expected Return", "Allocation": f"{performance[0]*100:.2f}%"},
        {"Ticker": "Volatility", "Allocation": f"{performance[1]*100:.2f}%"},
        {"Ticker": "Sharpe Ratio", "Allocation": f"{performance[2]*100:.2f}"},
        {"Ticker": "Dividend Yield", "Allocation": f"{total_div_yield*100:.2f}%"}
    ]

    df_metrics = pd.DataFrame(metrics)

    report = pd.concat([df_weights, df_metrics], ignore_index=True)
    return report.to_csv(index=False).encode('utf-8')