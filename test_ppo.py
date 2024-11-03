from stable_baselines3 import PPO
import pandas as pd
from portfolio_env import PortfolioEnv  # Import the custom environment
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Load the simple returns data
simple_returns = pd.read_csv('simple_returns.csv', index_col='Date')

# Initialize the custom environment
env = DummyVecEnv([lambda: PortfolioEnv(simple_returns, render_mode="human")])  # Add render_mode="human" for testing

# Load the trained model
model = PPO.load("ppo_portfolio_model")

# ---- Calculate Buy-and-Hold Strategy ----
# Equal weights for all assets (assuming equal initial investment in all)
buy_and_hold_weights = np.array([1.0 / simple_returns.shape[1]] * simple_returns.shape[1])

# Calculate cumulative returns for buy-and-hold strategy
# Using cumulative product for simple returns
buy_and_hold_cumulative_return = (simple_returns + 1).cumprod().dot(buy_and_hold_weights)

# ---- Test the trained PPO model ----
obs = env.reset()  # Reset the environment
ppo_cumulative_return = 1.0  # Starting cumulative return at 1 for the PPO strategy

for i in range(10):  # Simulate 1000 steps with the trained agent
    action, _states = model.predict(obs)

    # Perform a step in the environment
    obs, reward, done, info = env.step(action)

    # Access the first item in the info list
    portfolio_return = info[0]['portfolio_return']  # Retrieve actual portfolio return from the environment

    # Accumulate cumulative return for the PPO agent (based on portfolio returns, not rewards)
    ppo_cumulative_return *= (1 + portfolio_return)

    # Only print or render every 100 steps

    env.render()  # Render the environment (or print debugging info)
    print(f"PPO Strategy Cumulative Return: {ppo_cumulative_return}")
    print(f"Buy-and-Hold Cumulative Return: {buy_and_hold_cumulative_return.iloc[min(i, len(buy_and_hold_cumulative_return)-1)]}")


# Stop if the episode is over
    if done:
        break
