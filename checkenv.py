from stable_baselines3.common.env_checker import check_env
from portfolio_env import PortfolioEnv
import pandas as pd

data = pd.read_csv('preprocessed_stock_data.csv')

env = PortfolioEnv(data, window_size=63, horizon=252, max_steps=500)

check_env(env)