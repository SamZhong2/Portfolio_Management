import yfinance as yf
import pandas as pd

# List of tickers for the stocks you want to include in the portfolio
tickers = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'IBM']

# Download historical daily data for these tickers (5 years of data)
data = yf.download(tickers, start="2019-10-17", end="2024-10-17")

# Save adjusted closing prices, which account for splits/dividends
adj_close = data['Adj Close']

# Fill missing values (if any) and save to CSV
adj_close.fillna(method="ffill", inplace=True)
adj_close.to_csv('stock_data.csv')

# Display first few rows of the dataset
print(adj_close.head())
