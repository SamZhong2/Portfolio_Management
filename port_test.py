import os
import pandas as pd
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
import numpy as np

# Load the best model path
best_model_path = "models/1733607109/model_7000.zip"  # Update with the actual best model path

if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Best model file not found at: {best_model_path}")

# Load the dataset and create the test environment
data = pd.read_csv('preprocessed_stock_data.csv')
train_size = int(len(data) * 0.8)  # 80% for training
test_data = data.iloc[train_size:].reset_index(drop=True)
test_env = PortfolioEnv(test_data)

# Load the trained model
model = PPO.load(best_model_path)

obs, _ = test_env.reset()  # Reset the environment and get the initial observation
done = False
cumulative_reward = 0

while not done:
    
    # Get the model's allocation for the current observation
    action, _ = model.predict(obs)  
    allocation = (action + 1) / 2
    allocation = np.clip(allocation, 0, 1)  # Ensure valid weights
    allocation_sum = np.sum(allocation) + 1e-8  # Avoid division by zero
    allocation = allocation / allocation_sum  # Normalize to sum to 1
    print("Allocation: ", allocation)  # Log the allocation

    # Step through the environment
    obs, reward, terminated, truncated, _ = test_env.step(action)
    cumulative_reward += reward
    done = terminated or truncated

print(f"Total reward on test set: {cumulative_reward}")
