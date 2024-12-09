# Backtesting Script for a Specific Date
import pandas as pd
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Load the test data
data = pd.read_csv('preprocessed_stock_data.csv')
train_size = int(len(data) * 0.8)  # Use the last 20% for testing
# test_data = data.iloc[train_size:].reset_index(drop=True)
test_data = data

# Create the test environment
test_env = PortfolioEnv(test_data)

# Load the trained model
model_path = "models/1733754366/model_2257000.zip"  # Update this with the actual path
model = PPO.load(model_path)

# Select a specific date to test
specific_date = "2001-01-02"  # Replace with your desired date
specific_idx = test_data.index[test_data['Date'] == specific_date].tolist()

if not specific_idx:
    raise ValueError(f"Date {specific_date} not found in test data!")

specific_idx = specific_idx[0]

# Ensure there's enough data for the observation window and horizon
if specific_idx < test_env.window_size or specific_idx + test_env.horizon >= len(test_data):
    raise ValueError(f"Not enough data around the date {specific_date} for testing!")

# Prepare the observation (previous 30 days)
non_price_features = [col for col in test_data.columns[1:] if not col.endswith('_Price')]
observation = test_data[non_price_features].iloc[specific_idx - test_env.window_size : specific_idx].values.flatten()

# Get model's predicted allocation
action, _ = model.predict(observation)
action = np.clip(action, 0, 1)
action /= np.sum(action)  # Normalize to sum to 1
model_allocation = action

# Calculate equal weight allocation
num_assets = test_env.num_assets
equal_weights = np.ones(num_assets + 1) / (num_assets + 1)

# Simulate returns over the next 200 days
future_returns = test_data[[col for col in test_data.columns if col.endswith('_Return')]].iloc[
    specific_idx : specific_idx + test_env.horizon
].values

# Calculate portfolio returns for model and equal weight allocations
model_portfolio_returns = np.dot(future_returns, model_allocation[:-1]) + (model_allocation[-1] * test_env.risk_free_rate)
equal_weight_portfolio_returns = np.dot(future_returns, equal_weights[:-1]) + (equal_weights[-1] * test_env.risk_free_rate)

# Calculate cumulative returns
model_cumulative_return = np.prod(1 + model_portfolio_returns) - 1
equal_weight_cumulative_return = np.prod(1 + equal_weight_portfolio_returns) - 1

# Calculate Sharpe ratios (annualized)
model_mean_return = np.mean(model_portfolio_returns) * 252
model_volatility = np.std(model_portfolio_returns) * np.sqrt(252)
model_sharpe_ratio = (model_mean_return - test_env.risk_free_rate * 252) / (model_volatility + 1e-8)

equal_weight_mean_return = np.mean(equal_weight_portfolio_returns) * 252
equal_weight_volatility = np.std(equal_weight_portfolio_returns) * np.sqrt(252)
equal_weight_sharpe_ratio = (equal_weight_mean_return - test_env.risk_free_rate * 252) / (equal_weight_volatility + 1e-8)

# Print results
print(f"Testing Date: {specific_date}")
print(f"Model Allocation: {model_allocation}")
print(f"Equal Weight Allocation: {equal_weights}")
print(f"\nPerformance Metrics:")
print(f"Model Strategy: Sharpe Ratio: {model_sharpe_ratio:.4f}, Cumulative Return: {model_cumulative_return:.4f}")
print(f"Equal Weight Strategy: Sharpe Ratio: {equal_weight_sharpe_ratio:.4f}, Cumulative Return: {equal_weight_cumulative_return:.4f}")

# Plot cumulative returns
model_cumulative_values = (1 + np.cumsum(model_portfolio_returns))
equal_weight_cumulative_values = (1 + np.cumsum(equal_weight_portfolio_returns))

plt.figure(figsize=(12, 6))
plt.plot(range(len(model_cumulative_values)), model_cumulative_values, label='Model Strategy')
plt.plot(range(len(equal_weight_cumulative_values)), equal_weight_cumulative_values, label='Equal Weight Strategy')
plt.title(f"Portfolio Performance Starting {specific_date}")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid()
plt.show()
