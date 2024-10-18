import pandas as pd

# Load the saved stock data (adjusted closing prices)
adj_close = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Calculate daily simple returns (percentage change between consecutive days)
simple_returns = adj_close.pct_change().dropna()

# Save simple returns to a CSV file for future use
simple_returns.to_csv('simple_returns.csv')

# Display the first few rows to ensure everything looks correct
print(simple_returns.head())
