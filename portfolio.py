import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

#Fetch historical data from Yahoo Finance for Apple and Tesla
assets = ['AAPL', 'TSLA']
data = yf.download(assets, start='2023-01-01', end='2026-01-01')['Close']

returns = data.pct_change().dropna() # computes % change day-by-day

moving_avg_50 = data.rolling(window=50).mean()#moving averag calculation for trend identification
moving_avg_200 = data.rolling(window=200).mean()

weights = [0.6, 0.4]  # Portfolio weights
portfolio_returns = (returns * weights).sum(axis=1)#daily combined portfolio returns
portfolio_cumulative = (1 + portfolio_returns).cumprod()#how portfolio value grows over time


expected_return = np.mean(portfolio_returns) * 252  # Annualized average return
volatility = np.std(portfolio_returns) * np.sqrt(252)#measure of risks

#Vizualize the portfolio
plt.figure(figsize=(10,6))
plt.plot(portfolio_cumulative, label='Portfolio Value')
plt.title('Simulated Portfolio Growth')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()