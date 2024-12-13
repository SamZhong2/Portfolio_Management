# Test some parameters for the PPO model, no hyperparameter tuning
# 1. Test 252/4 days behind the current date with horizon of 252
# 2. Test 252/4 days behind the current date with horizon of 252/4
# 3. Test 252/12 days behind the current date with horizon of 252/4


import time
import pandas as pd
import os
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

# Setup directories
models_dir = f"models/month-quarter"
logdir = f"logs/month-quarter/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Load data
data = pd.read_csv('preprocessed_stock_data.csv')
train_size = int(len(data) * 0.7)
train_data = data.iloc[:train_size].reset_index(drop=True)
test_data = data.iloc[train_size:].reset_index(drop=True)

# Environments
train_env = PortfolioEnv(train_data, window_size=63, horizon=252, max_steps=500)
validation_env = PortfolioEnv(test_data, window_size=63, horizon=252, max_steps=50)

# Initialize model and logger
model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=logdir)
writer = SummaryWriter(logdir)

# Training settings
TIMESTEPS = 10000
iters = 0
best_reward = -float('inf')
best_model_path = None
patience = 10
no_improve_count = 0
previous_model_path = None

def evaluate_model(env, model):
    obs, _ = env.reset()
    done = False
    cumulative_rewards = []
    sharpe_ratios = []
    sortino_ratios = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_rewards.append(info['cumulative_return'])
        sharpe_ratios.append(info['sharpe_ratio'])
        sortino_ratios.append(info['sortino_ratio'])
        done = terminated or truncated

    return {
        'avg_cumulative_return': sum(cumulative_rewards) / len(cumulative_rewards),
        'avg_sharpe_ratio': sum(sharpe_ratios) / len(sharpe_ratios),
        'avg_sortino_ratio': sum(sortino_ratios) / len(sortino_ratios),
        'reward': reward
    }

# Training loop
# Training loop
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")

    # Save the current model
    model_path = f"{models_dir}/model_{TIMESTEPS * iters}.zip"
    model.save(model_path)

    # Evaluate on training and validation environments
    train_results = evaluate_model(train_env, model)
    validation_results = evaluate_model(validation_env, model)

    # Log metrics
    writer.add_scalar("Training/Average Cumulative Return", train_results['avg_cumulative_return'], iters)
    writer.add_scalar("Training/Average Sharpe Ratio", train_results['avg_sharpe_ratio'], iters)
    writer.add_scalar("Training/Average Sortino Ratio", train_results['avg_sortino_ratio'], iters)
    writer.add_scalar("Validation/Average Cumulative Return", validation_results['avg_cumulative_return'], iters)
    writer.add_scalar("Validation/Average Sharpe Ratio", validation_results['avg_sharpe_ratio'], iters)
    writer.add_scalar("Validation/Average Sortino Ratio", validation_results['avg_sortino_ratio'], iters)

    current_reward = validation_results['reward']

    if current_reward > best_reward:
        best_reward = current_reward
        best_model_path = f"{models_dir}/best_model.zip"

        # Save the new best model
        model.save(best_model_path)

        # If there was a previously saved "best" model, delete it
        if previous_model_path and previous_model_path != best_model_path:
            os.remove(previous_model_path)

        # Update the previous model path
        previous_model_path = model_path

        no_improve_count = 0  # Reset patience
    else:
        no_improve_count += 1
        # Delete non-best models to save space
        if os.path.exists(model_path):
            os.remove(model_path)

    # Print results
    print(f"Iteration {iters}:")
    print(f"  Training Avg Sharpe Ratio: {train_results['avg_sharpe_ratio']}")
    print(f"  Validation Avg Sharpe Ratio: {validation_results['avg_sharpe_ratio']}")
    print(f"  Best Reward Model: {best_model_path} with Reward: {best_reward}")

# Close the writer
writer.close()
