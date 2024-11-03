import yfinance as yf
import pandas as pd

# List of 10 diversified tickers from different sectors
tickers = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 'PG', 'NEE', 'KO', 'NVDA', 'AMZN']

# Download historical weekly data for these tickers (20 years of data)
data = yf.download(tickers, start="2004-09-01", end="2024-09-01", interval='1wk')

# Extract the adjusted close prices, which account for dividends and stock splits
adj_close = data['Adj Close']

# Fill missing values (forward fill and backward fill) to handle any gaps in the data
adj_close.fillna(method="ffill", inplace=True)
adj_close.fillna(method="bfill", inplace=True)

# Calculate weekly returns for all the tickers
weekly_returns = adj_close.pct_change().dropna()

# Save the weekly returns to a CSV file for use in PPO model
weekly_returns.to_csv('weekly_stock_returns.csv')

# Display the first few rows of the dataset to confirm the data
print(weekly_returns.head())
