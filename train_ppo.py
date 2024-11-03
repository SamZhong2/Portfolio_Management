from stable_baselines3 import PPO
import pandas as pd
from portfolio_env import PortfolioEnv  # Import the custom environment
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Load the simple returns data
simple_returns = pd.read_csv('simple_returns.csv', index_col='Date')

# Initialize the custom environment with simple returns
env = DummyVecEnv([lambda: PortfolioEnv(simple_returns)])  # No need for render_mode in training

# Initialize the evaluation environment
eval_env = DummyVecEnv([lambda: PortfolioEnv(simple_returns)])

# Initialize the PPO agent with the MLP (Multilayer Perceptron) policy
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

# Create an evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False)

# Train the agent for a specified number of timesteps, using the evaluation callback
model.learn(total_timesteps=500000, callback=eval_callback)

# Save the final trained model
model.save("ppo_portfolio_model")

# Optionally, save the last evaluation best model
model.save("./logs/best_model/ppo_portfolio_best")
