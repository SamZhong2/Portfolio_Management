import yfinance as yf
import pandas as pd

# List of 10 diversified tickers from different sectors
tickers = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 'PG', 'NEE', 'KO', 'NVDA', 'AMZN']

# Download historical daily data for these tickers
data = yf.download(tickers, start="2004-09-01", end="2024-09-01", interval="1d")

# Extract the adjusted close prices, which account for dividends and stock splits
adj_close = data['Adj Close']

# Fill missing values (forward fill and backward fill) to handle any gaps in the data
adj_close.fillna(method="ffill", inplace=True)
adj_close.fillna(method="bfill", inplace=True)

# Drop timezone information from the index (make it timezone-naive)
adj_close.index = adj_close.index.tz_localize(None)

# Save the daily adjusted close prices to a CSV file for use in PPO model
adj_close.to_csv('daily_stock_prices.csv')

# Display the first few rows of the dataset to confirm the data
print(adj_close.head())
