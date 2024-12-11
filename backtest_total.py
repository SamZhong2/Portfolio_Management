import pandas as pd
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import torch


# Load the data
data = pd.read_csv('preprocessed_stock_data.csv')

# Load the trained model
model_path = "models/quarter-year/best_model.zip"  # Update this with the actual path
model = PPO.load(model_path)

# Ensure SPY_Return is in the data
if 'SPY_Simple_Return' not in data.columns:
    raise ValueError("SPY_Simple_Return column not found in the dataset!")

# Filter the data from 2016 to 2023
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'].dt.year >= 2016) & (data['Date'].dt.year <= 2023)]

# Initialize variables
data['Year'] = data['Date'].dt.year
years = sorted(data['Year'].unique())
cumulative_results = {
    'Year': [],
    'Model Cumulative Return': [],
    'Equal Weight Cumulative Return': [],
    'SPY Cumulative Return': []
}

model_cumulative_value = 1  # Start with 1 for cumulative calculations
equal_weight_cumulative_value = 1
spy_cumulative_value = 1

# # Iterate over each year
# for year in years:
#     # Filter data for the specific year
#     year_data = data[data['Year'] == year].reset_index(drop=True)

#     # Skip years with insufficient data
#     if len(year_data) < 30:  # Assuming a 30-day observation window
#         continue

#     # Prepare the observation on the first day of the year
#     test_env = PortfolioEnv(year_data)
#     non_price_features = [col for col in year_data.columns[1:-1] if not col.endswith('_Price')]
#     observation = year_data[non_price_features].iloc[:test_env.window_size].values.flatten()

#     # Normalize observations (min-max scaling)
#     obs_min = np.min(observation, axis=0)
#     obs_max = np.max(observation, axis=0)
#     normalized_obs = (observation - obs_min) / (obs_max - obs_min + 1e-8)

#     # Get model's predicted allocation
#     action, _ = model.predict(normalized_obs)
#     action = (action + 1) / 2  # Rescale to [0, 1]
#     action = np.clip(action, 0, 1)
#     action /= np.sum(action)  # Normalize to sum to 1

#     # Equal weight allocation
#     equal_weights = np.ones(test_env.num_assets + 1) / (test_env.num_assets + 1)

#     # Simulate returns for the year
#     future_returns = year_data[[col for col in year_data.columns if col.endswith('_Return')]].values
#     model_portfolio_returns = np.dot(future_returns, action[:-1]) + (action[-1] * test_env.risk_free_rate)
#     equal_weight_portfolio_returns = np.dot(future_returns, equal_weights[:-1]) + (equal_weights[-1] * test_env.risk_free_rate)
#     spy_returns = year_data['SPY_Simple_Return'].values

#     # Update cumulative returns
#     model_cumulative_value *= np.prod(1 + model_portfolio_returns)
#     equal_weight_cumulative_value *= np.prod(1 + equal_weight_portfolio_returns)
#     spy_cumulative_value *= np.prod(1 + spy_returns)

#     # Store results
#     cumulative_results['Year'].append(year)
#     cumulative_results['Model Cumulative Return'].append(model_cumulative_value - 1)
#     cumulative_results['Equal Weight Cumulative Return'].append(equal_weight_cumulative_value - 1)
#     cumulative_results['SPY Cumulative Return'].append(spy_cumulative_value - 1)

# # Convert results to DataFrame
# cumulative_results_df = pd.DataFrame(cumulative_results)

num_samples = 100
all_results = []

for sample in range(num_samples):
    # Set a random seed for reproducibility
    np.random.seed(sample)
    torch.manual_seed(sample)

    # Reinitialize cumulative values for each sample
    model_cumulative_value = 1
    equal_weight_cumulative_value = 1
    spy_cumulative_value = 1

    # Temporary dictionary for this sample's results
    cumulative_results = {
        'Year': [],
        'Model Cumulative Return': [],
        'Equal Weight Cumulative Return': [],
        'SPY Cumulative Return': []
    }

 

    # Iterate over each year
    for year in years:
        # Filter data for the specific year
        year_data = data[data['Year'] == year].reset_index(drop=True)

        # Skip years with insufficient data
        if len(year_data) < 30:  # Assuming a 30-day observation window
            continue

        # Prepare the observation on the first day of the year
        test_env = PortfolioEnv(year_data)
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

        # Simulate returns for the year
        future_returns = year_data[[col for col in year_data.columns if col.endswith('_Return')]].values
        model_portfolio_returns = np.dot(future_returns, action[:-1]) + (action[-1] * test_env.risk_free_rate)
        equal_weight_portfolio_returns = np.dot(future_returns, equal_weights[:-1]) + (equal_weights[-1] * test_env.risk_free_rate)
        spy_returns = year_data['SPY_Simple_Return'].values

        # Update cumulative returns
        model_cumulative_value *= np.prod(1 + model_portfolio_returns)
        equal_weight_cumulative_value *= np.prod(1 + equal_weight_portfolio_returns)
        spy_cumulative_value *= np.prod(1 + spy_returns)

        # Store results
        cumulative_results['Year'].append(year)
        cumulative_results['Model Cumulative Return'].append(model_cumulative_value - 1)
        cumulative_results['Equal Weight Cumulative Return'].append(equal_weight_cumulative_value - 1)
        cumulative_results['SPY Cumulative Return'].append(spy_cumulative_value - 1)

    # Append this sample's results to the aggregated results
    cumulative_results_df = pd.DataFrame(cumulative_results)
    all_results.append(cumulative_results_df)

# Calculate the average results across all samples
average_results = pd.concat(all_results).groupby('Year').mean().reset_index()

# Plot the averaged cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(average_results['Year'], average_results['Model Cumulative Return'], label='Model Cumulative Return (Avg)')
plt.plot(average_results['Year'], average_results['Equal Weight Cumulative Return'], label='Equal Weight Cumulative Return (Avg)')
plt.plot(average_results['Year'], average_results['SPY Cumulative Return'], label='SPY Cumulative Return (Avg)', linestyle='--')
plt.title("Averaged Cumulative Returns (2016-2023, 10 Samples)")
plt.xlabel("Year")
plt.ylabel("Averaged Cumulative Return")
plt.legend()
plt.grid()
plt.show()

# Print the averaged DataFrame for analysis
print(average_results)