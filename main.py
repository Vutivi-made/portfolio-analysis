import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# User inputs for tickers, date window, and weights
raw = input("Enter ticker symbols separated by commas (e.g. AAPL, TSLA): ")
assets = [t.strip().upper() for t in raw.split(',')]
start = input("Enter start date (YYYY-MM-DD): ")
end = input("Enter end date (YYYY-MM-DD): ")
raw_weights = input(f"Enter weights for {assets} separated by commas (must sum to 1, e.g. 0.5, 0.3, 0.2): ")
weights = [float(w.strip()) for w in raw_weights.split(',')]

# Fetch historical data from Yahoo Finance
data = yf.download(assets, start=start, end=end, auto_adjust=True)['Close']
returns = data.pct_change().dropna() # computes % change day-by-day
moving_avg_50 = data.rolling(window=50).mean() # moving average calculation for trend identification
moving_avg_200 = data.rolling(window=200).mean()
portfolio_returns = (returns * weights).sum(axis=1) # daily combined portfolio returns
portfolio_cumulative = (1 + portfolio_returns).cumprod() # how portfolio value grows over time
expected_return = np.mean(portfolio_returns) * 252 # annualized average return
volatility = np.std(portfolio_returns) * np.sqrt(252) # measure of risk

# Visualize the portfolio
plt.figure(figsize=(10,6))
plt.plot(portfolio_cumulative, label='Portfolio Value')
plt.title('Simulated Portfolio Growth')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()