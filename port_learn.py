import time
import pandas as pd
import os
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

data = pd.read_csv('preprocessed_stock_data.csv')

env = PortfolioEnv(data)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
