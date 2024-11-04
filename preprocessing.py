import pandas as pd
import numpy as np
from fredapi import Fred

# Initialize FRED client
fred = Fred(api_key='dbc938743655482a0f341e0216a5c36e')

# Fetch Federal Funds Effective Rate
interest_rate = fred.get_series('FEDFUNDS', start_date='2004-09-01', end_date='2024-09-01')

# Fetch Unemployment Rate
unemployment_rate = fred.get_series('UNRATE', start_date='2004-09-01', end_date='2024-09-01')

# Convert to DataFrame
interest_rate_df = interest_rate.to_frame(name='Interest_Rate')
unemployment_rate_df = unemployment_rate.to_frame(name='Unemployment_Rate')

# Load your daily stock data
daily_stock_data = pd.read_csv('daily_stock_prices.csv', index_col='Date', parse_dates=True)

# Step 1: Calculate Adjusted Log Return
# Adjusted log return: log of today's close price over yesterday's close price
log_returns = np.log(daily_stock_data / daily_stock_data.shift(1))
log_returns.columns = [f"{col}_Log_Return" for col in log_returns.columns]


# Step 2: Calculate Daily RSI (14-day)
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


rsi_data = daily_stock_data.apply(lambda x: calculate_rsi(x), axis=0)
rsi_data.columns = [f"{col}_RSI" for col in rsi_data.columns]

# Step 3: Calculate Moving Averages on Daily Data
ma_20 = daily_stock_data.rolling(window=20).mean().add_suffix('_20d_MA')
ma_60 = daily_stock_data.rolling(window=60).mean().add_suffix('_60d_MA')

# Step 4: Add Interest Rate and Unemployment Rate (Daily Frequency or Resampled to Match if Available)
# Placeholder data (replace with actual interest rate and unemployment rate data if available)
date_range = daily_stock_data.index
interest_rate = pd.DataFrame(0.05, index=date_range, columns=['Interest_Rate'])  # Example 5%
unemployment_rate = pd.DataFrame(0.04, index=date_range, columns=['Unemployment_Rate'])  # Example 4%

preprocessed_daily_data = pd.concat([daily_stock_data.add_suffix('_Price'),
                                     log_returns,
                                     rsi_data,
                                     ma_20,
                                     ma_60,
                                     interest_rate,
                                     unemployment_rate], axis=1)

# Drop any rows with NaN values from rolling calculations
preprocessed_daily_data.dropna(inplace=True)

# Step 5: Select Weekly Reallocation Days (e.g., Every Friday)
# Create a mask for selecting Fridays
weekly_reallocation_days = preprocessed_daily_data.index.weekday == 4  # Friday is weekday 4

# Filter Data for Weekly Reallocation
weekly_data_for_reallocation = preprocessed_daily_data[weekly_reallocation_days]

# Save both full daily data and weekly reallocation points
preprocessed_daily_data.to_csv('preprocessed_daily_stock_data.csv')
weekly_data_for_reallocation.to_csv('weekly_reallocation_points.csv')

# Display the first few rows to confirm structure
print("Daily Data Preview:")
print(preprocessed_daily_data.head())
print("\nWeekly Reallocation Points Preview:")
print(weekly_data_for_reallocation.head())
