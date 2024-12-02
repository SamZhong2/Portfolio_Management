from portfolio_env import PortfolioEnv
import pandas as pd

data = pd.read_csv('preprocessed_stock_data.csv')

env = PortfolioEnv(data)
eps = 1

for eps in range(eps):
    terminated = False
    obs, _ = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        random_action = (random_action + 1) / 2
        print("action: ", random_action)
        obs, reward, terminated, truncated, _ = env.step(random_action)
        print("reward: ", reward)


