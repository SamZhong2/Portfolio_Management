import pandas as pd
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import torch

# Load the data
data = pd.read_csv('test_preprocessed_stock_data.csv')

# Load the trained model
model_path = "models/quarter-year/best_model.zip"  # Update this with the actual path
model = PPO.load(model_path)

# Ensure SPY_Return is in the data
if 'SPY_Simple_Return' not in data.columns:
    raise ValueError("SPY_Return column not found in the dataset!")

# Initialize variables
data['Year'] = pd.to_datetime(data['Date']).dt.year
years = data['Year'].unique()
annual_results = []

# Iterate over each year
for year in years:
    # Filter data for the specific year
    year_data = data[data['Year'] == year].reset_index(drop=True)

    # Prepare the observation on the first day of the year
    test_env = PortfolioEnv(year_data,  window_size=63, horizon=252, max_steps=500)
    non_price_features = [col for col in year_data.columns[1:-1] if not col.endswith('_Price')]
    observation = year_data[non_price_features].iloc[:test_env.window_size].values.flatten()

    # Normalize observations (min-max scaling)
    obs_min = np.min(observation, axis=0)
    obs_max = np.max(observation, axis=0)
    normalized_obs = (observation - obs_min) / (obs_max - obs_min + 1e-8)

    # Get model's predicted allocation
    action, _ = model.predict(normalized_obs)
    action = (action + 1) / 2  # Rescale to [0, 1]
    action = np.clip(action, 0, 1)
    action /= np.sum(action)  # Normalize to sum to 1

    # Equal weight allocation
    equal_weights = np.ones(test_env.num_assets + 1) / (test_env.num_assets + 1)
    equal_weights_no_cash = np.ones(test_env.num_assets) / test_env.num_assets

    # Simulate returns for the year
    future_returns = year_data[[col for col in year_data.columns if col.endswith('_Return')]].values
    model_portfolio_returns = np.dot(future_returns, action[:-1]) + (action[-1] * test_env.risk_free_rate)
    equal_weight_portfolio_returns = np.dot(future_returns, equal_weights[:-1]) + (equal_weights[-1] * test_env.risk_free_rate)
    equal_weight_portfolio_returns_no_cash = np.dot(future_returns, equal_weights_no_cash)
    spy_returns = year_data['SPY_Simple_Return'].values

    # Calculate cumulative returns
    model_cumulative_return = np.prod(1 + model_portfolio_returns) - 1
    equal_weight_cumulative_return = np.prod(1 + equal_weight_portfolio_returns) - 1
    equal_weight_cumulative_return_no_cash = np.prod(1 + equal_weight_portfolio_returns_no_cash) - 1
    spy_cumulative_return = np.prod(1 + spy_returns) - 1

    # Store annual results
    annual_results.append({
        'Year': year,
        'Model Return': model_cumulative_return,
        'Equal Weight Return': equal_weight_cumulative_return,
        'Equal Weight Return (No Cash)': equal_weight_cumulative_return_no_cash,
        'SPY Return': spy_cumulative_return
    })

# Convert to DataFrame
annual_results_df = pd.DataFrame(annual_results)

def smooth_data(values, window_size=2):
     return pd.Series(values).rolling(window=window_size, min_periods=1).mean()

annual_results_df['Smoothed Model Return'] = smooth_data(annual_results_df['Model Return'])
annual_results_df['Smoothed SPY Return'] = smooth_data(annual_results_df['SPY Return'])
annual_results_df['Smoothed Equal Weight Return'] = smooth_data(annual_results_df['Equal Weight Return'])

# Plot the annual returns
plt.figure(figsize=(12, 6))
plt.plot(annual_results_df['Year'], annual_results_df['Smoothed Model Return'], label='Smoothed Model Return')
plt.plot(annual_results_df['Year'], annual_results_df['Smoothed SPY Return'], label='Smoothed SPY Return', linestyle='--')
plt.title("Annual Portfolio Returns Comparison")
plt.xlabel("Year")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(annual_results_df['Year'], annual_results_df['Smoothed Model Return'], label='Smoothed Model Return')
plt.plot(annual_results_df['Year'], annual_results_df['Smoothed Equal Weight Return'], label='Smoothed Equal Weight Return')
plt.title("Annual Portfolio Returns Comparison")
plt.xlabel("Year")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid()
plt.show()

# Print the DataFrame for analysis
print(annual_results_df)
