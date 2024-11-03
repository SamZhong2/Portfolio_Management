import yfinance as yf
import pandas as pd
from fredapi import Fred
import numpy as np

# API Key for FRED (you need to sign up at https://fred.stlouisfed.org/ to get an API key)
fred = Fred(api_key='dbc938743655482a0f341e0216a5c36e')

# Load the weekly stock returns from the local CSV file
weekly_returns = pd.read_csv('weekly_stock_returns.csv', index_col='Date', parse_dates=True)

# Ensure that the weekly stock returns are aligned by forward-filling and backward-filling any missing data
weekly_returns = weekly_returns.ffill().bfill()

# Remove any timezone information
weekly_returns.index = weekly_returns.index.tz_localize(None)

# Calculate moving averages (MA) and rolling standard deviations (volatility) if not already calculated
window_size = 4
moving_avg = weekly_returns.rolling(window=window_size).mean()
rolling_std = weekly_returns.rolling(window=window_size).std()

# Drop NaNs in moving averages and standard deviation
moving_avg.dropna(inplace=True)
rolling_std.dropna(inplace=True)

# Rename stock-specific columns to include stock tickers
weekly_returns.columns = [f'{stock}_returns' for stock in weekly_returns.columns]
moving_avg.columns = [f'{stock}_4w_ma' for stock in moving_avg.columns]
rolling_std.columns = [f'{stock}_4w_vol' for stock in rolling_std.columns]

# Collect macroeconomic data from FRED
cpi_data = fred.get_series('CPIAUCSL', observation_start='2004-09-01', observation_end='2024-09-01').resample('W-MON').ffill()
interest_rate = fred.get_series('FEDFUNDS', observation_start='2004-09-01', observation_end='2024-09-01').resample('W-MON').ffill()
unemployment_rate = fred.get_series('UNRATE', observation_start='2004-09-01', observation_end='2024-09-01').resample('W-MON').ffill()

# Download the VIX index (market volatility) using yfinance
vix = yf.download('^VIX', start="2004-09-01", end="2024-09-01", interval='1wk')['Adj Close']
vix = vix.ffill().bfill()

# Combine all macroeconomic indicators into one DataFrame
macro_data = pd.DataFrame({
    'CPI': cpi_data,
    'FedFundsRate': interest_rate,
    'UnemploymentRate': unemployment_rate,
    'VIX': vix
})

# Forward-fill and backward-fill missing values in the macro data
macro_data = macro_data.ffill().bfill()

# Align all DataFrames by index (ensure they have the same index)
common_index = weekly_returns.index.intersection(macro_data.index)

# Reindex all DataFrames to ensure alignment
weekly_returns = weekly_returns.reindex(common_index)
moving_avg = moving_avg.reindex(common_index)
rolling_std = rolling_std.reindex(common_index)
macro_data = macro_data.reindex(common_index)

# Now get the correct number of stocks
num_stocks = len(weekly_returns.columns)  # Number of stocks

# Stack stock-specific features together into a 3D array (time_steps, num_stocks, num_features_per_stock)
stock_features = np.stack([weekly_returns.values, moving_avg.values, rolling_std.values], axis=-1)

# Now, macroeconomic data has to be repeated across each stock for each time step
macro_data_repeated = np.repeat(macro_data.values[:, np.newaxis, :], num_stocks, axis=1)

# Combine stock-specific features with macroeconomic indicators
# Concatenating along the last axis to ensure macro data is treated as additional features
combined_data = np.concatenate([stock_features, macro_data_repeated], axis=-1)

# The shape should now be (time_steps, num_stocks, features_per_stock + macro_indicators)
print(combined_data.shape)

# Optionally save the final reshaped data to a CSV or .npy file if needed
np.save('final_stock_and_macro_data.npy', combined_data)


