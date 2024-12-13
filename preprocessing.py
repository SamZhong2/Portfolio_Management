import pandas as pd
import numpy as np
from fredapi import Fred

# Initialize FRED client with your API key (replace with your actual API key)
fred = Fred(api_key='dbc938743655482a0f341e0216a5c36e')

# Fetch Federal Funds Effective Rate and Unemployment Rate from FRED
interest_rate = fred.get_series('FEDFUNDS', start="1996-09-01", end_date='2024-12-01')
unemployment_rate = fred.get_series('UNRATE', start="1996-09-01", end_date='2024-12-01')

# Convert FRED data to DataFrames
interest_rate_df = interest_rate.to_frame(name='Interest_Rate')
unemployment_rate_df = unemployment_rate.to_frame(name='Unemployment_Rate')

# Load the lagged stock data and volume data from provided CSV files
daily_stock_data = pd.read_csv('daily_stock_prices.csv', index_col='Date', parse_dates=True)
daily_volume_data = pd.read_csv('daily_stock_volume.csv', index_col='Date', parse_dates=True)

# Step 1: Calculate Simple Return
simple_returns = daily_stock_data.pct_change().add_suffix('_Simple_Return')

print(simple_returns.head())

# Step 2: Calculate Daily RSI (14-day)
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


rsi_data = daily_stock_data.apply(lambda x: calculate_rsi(x), axis=0)
rsi_data.columns = [f"{col}_RSI" for col in rsi_data.columns]

# Step 3: Calculate Moving Averages on Daily Data with Three Different Intervals
ma_20 = daily_stock_data.rolling(window=20).mean().add_suffix('_20d_MA')
ma_50 = daily_stock_data.rolling(window=50).mean().add_suffix('_50d_MA')
ma_100 = daily_stock_data.rolling(window=100).mean().add_suffix('_100d_MA')

# Step 4: Resample Interest Rate and Unemployment Rate to Daily Frequency
interest_rate_daily = interest_rate_df.reindex(daily_stock_data.index).ffill().bfill()
unemployment_rate_daily = unemployment_rate_df.reindex(daily_stock_data.index).ffill().bfill()

# Step 5: Concatenate Data for Model Input, including volume data
preprocessed_daily_data = pd.concat([daily_stock_data.add_suffix('_Price'),
                                     daily_volume_data.add_suffix('_Volume'),
                                     simple_returns,
                                     rsi_data,
                                     ma_20,
                                     ma_50,
                                     ma_100,
                                     interest_rate_daily,
                                     unemployment_rate_daily], axis=1)

# Drop any rows with NaN values from rolling calculations
preprocessed_daily_data.dropna(inplace=True)

# Save both full daily data and weekly reallocation points
preprocessed_daily_data.to_csv('preprocessed_stock_data.csv')

# Display the first few rows to confirm structure
print("Daily Data Preview:")
print(preprocessed_daily_data.head())
