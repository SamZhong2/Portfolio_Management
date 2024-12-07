from portfolio_env import PortfolioEnv
import pandas as pd
import numpy as np

data = pd.read_csv('preprocessed_stock_data.csv')

env = PortfolioEnv(data)
eps = 1

for eps in range(eps):
    terminated = False
    obs, _ = env.reset()
    while not terminated:
        action = env.action_space.sample()
        action = (action + 1) / 2  # Rescale to [0, 1]
        action = np.clip(action, 0, 1)  # Ensure valid weights
        action_sum = np.sum(action) + 1e-8  # Avoid division by zero
        action = action / action_sum  # Normalize to sum to 1
        print("action: ", action)
        obs, reward, terminated, truncated, _ = env.step(action)
        print("reward: ", reward)


