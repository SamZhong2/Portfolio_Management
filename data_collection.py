import yfinance as yf
import pandas as pd

# List of 10 diversified tickers from different sectors
tickers = ['AAPL', 'JPM', 'MSFT']

# Download historical daily data for these tickers
data = yf.download(tickers, start="1990-09-01", end="2024-09-01", interval="1d")

# Extract the adjusted close prices and volume
adj_close = data['Adj Close']
volume = data['Volume']

# Fill missing values (forward fill and backward fill) to handle any gaps in the data
adj_close.fillna(method="ffill", inplace=True)
adj_close.fillna(method="bfill", inplace=True)

volume.fillna(method="ffill", inplace=True)
volume.fillna(method="bfill", inplace=True)

# Drop timezone information from the index (make it timezone-naive)
adj_close.index = adj_close.index.tz_localize(None)
volume.index = volume.index.tz_localize(None)

# Save the lagged adjusted close prices and volume data to CSV files for later use
adj_close.to_csv('daily_stock_prices.csv')
volume.to_csv('daily_stock_volume.csv')

# Display the first few rows of both lagged datasets to confirm the data
print("Lagged Adjusted Close Prices Preview:")
print(adj_close.head())

print("\nLagged Volume Data Preview:")
print(volume.head())
