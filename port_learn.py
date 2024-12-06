# import time
# import pandas as pd
# import os
# from portfolio_env import PortfolioEnv
# from stable_baselines3 import PPO

# models_dir = f"models/{int(time.time())}"
# logdir = f"logs/{int(time.time())}/"

# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

# if not os.path.exists(logdir):
#     os.makedirs(logdir)

# data = pd.read_csv('preprocessed_stock_data.csv')

# env = PortfolioEnv(data)

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 1000
# iters = 0
# while True:
#     iters += 1
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
#     model.save(f"{models_dir}/{TIMESTEPS * iters}")

from stable_baselines3.common.env_util import DummyVecEnv
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
import pandas as pd
import os
import time

# Create directories for models and logs
models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Load and split data
data = pd.read_csv('preprocessed_stock_data.csv')
train_data = data[:int(len(data) * 0.8)]  # 80% training
test_data = data[int(len(data) * 0.8):]   # 20% testing

# Wrap environments with DummyVecEnv for SB3 compatibility
train_env = DummyVecEnv([lambda: PortfolioEnv(train_data)])
test_env = DummyVecEnv([lambda: PortfolioEnv(test_data)])

# Initialize the model
model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=logdir)

# Training parameters
TIMESTEPS = 1000
iters = 0
best_reward = float('-inf')
best_model_path = None

# Training loop
while True:
    iters += 1
    # Train the model
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
    
    # Evaluate the model on the test set
    obs = test_env.reset()
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        rewards.append(reward)

    total_reward = sum(rewards)
    print(f"Iteration {iters}, Total Reward: {total_reward}")
    
    # Save the best model
    if total_reward > best_reward:
        best_reward = total_reward
        best_model_path = f"{models_dir}/best_model"
        model.save(best_model_path)
        print(f"New best model saved with reward: {best_reward}")

    # Optional: Stop training after certain iterations or criteria
    if iters >= 10:  # Example: stop after 50 iterations
        print("Stopping training.")
        break

# Test the best model
print(f"Loading best model from {best_model_path}")
best_model = PPO.load(best_model_path)

obs = test_env.reset()
rewards = []
done = False
while not done:
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)
    rewards.append(reward)

test_reward = sum(rewards)
print(f"Best Model Test Reward: {test_reward}")
