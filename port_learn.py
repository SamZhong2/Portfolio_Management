import time
import pandas as pd
import os
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

# Setup directories for saving models and logs
models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Load data
data = pd.read_csv('preprocessed_stock_data.csv')
train_size = int(len(data) * 0.8)  # 80% for training
train_data = data.iloc[:train_size].reset_index(drop=True)
test_data = data.iloc[train_size:].reset_index(drop=True)

# Create training environment
train_env = PortfolioEnv(train_data)

# Initialize model
model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=logdir)

# Training settings
TIMESTEPS = 1000
iters = 0
best_reward = -float('inf')
best_model_path = None

# Infinite training loop
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")

    # Save the model for this iteration
    model_path = f"{models_dir}/model_{TIMESTEPS * iters}.zip"
    model.save(model_path)

    # Evaluate the model on the training environment to get average metrics
    obs, _ = train_env.reset()  # Extract only the observation
    done = False
    cumulative_rewards = []
    sharpe_ratios = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = train_env.step(action)
        cumulative_rewards.append(info['cumulative_return'])
        sharpe_ratios.append(info['sharpe_ratio'])
        done = terminated or truncated

    # Compute averages
    avg_cumulative_return = sum(cumulative_rewards) / len(cumulative_rewards)
    avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios)

    # Keep track of the best model
    if avg_sharpe_ratio > best_reward:
        best_reward = avg_sharpe_ratio
        best_model_path = model_path

    # Log results for the iteration
    print(f"Iteration {iters}:")
    print(f"  Average Cumulative Return: {avg_cumulative_return}")
    print(f"  Average Sharpe Ratio: {avg_sharpe_ratio}")
    print(f"  Best Reward Model: {best_model_path} with Reward: {best_reward}")
