from stable_baselines3.common.env_checker import check_env
from portfolio_env import PortfolioEnv
import pandas as pd

data = pd.read_csv('preprocessed_stock_data.csv')

env = PortfolioEnv(data)

check_env(env)