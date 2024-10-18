import gymnasium as gym  # Use gymnasium instead of gym
from gymnasium import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    def __init__(self, simple_returns, render_mode=None):
        super(PortfolioEnv, self).__init__()
        self.num_assets = simple_returns.shape[1]
        self.simple_returns = simple_returns.values  # Use simple returns
        self.n_steps = self.simple_returns.shape[0]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets + 1,), dtype=np.float32)
        self.weights = np.array([1.0 / self.num_assets] * self.num_assets)
        self.current_step = 0
        self.cumulative_return = 1  # Track cumulative returns directly now
        self.returns_history = []
        self.peak_value = 1  # Track peak portfolio value for drawdown calculation
        self.render_mode = render_mode  # Add render mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.weights = np.array([1.0 / self.num_assets] * self.num_assets, dtype=np.float32)
        self.current_step = 0
        self.cumulative_return = 1  # Reset cumulative return to 1
        self.returns_history = []
        self.peak_value = 1
        observation = np.concatenate(([self.weights.sum()], self.simple_returns[self.current_step])).astype(np.float32)
        return observation, {}

    def step(self, action):
        # Clip and normalize weights
        weights = np.clip(action, 0, 1)
        weight_sum = np.sum(weights)

        # If weights sum to zero (which can happen if the agent chooses all zeros), we set equal weights
        if weight_sum == 0:
            weights = np.array([1.0 / self.num_assets] * self.num_assets, dtype=np.float32)
        else:
            weights /= weight_sum  # Normalize the weights so they sum to 1

        # Calculate the portfolio return based on the current asset simple returns
        portfolio_return = np.dot(weights, self.simple_returns[self.current_step])

        # Update cumulative return directly
        self.cumulative_return *= (1 + portfolio_return)

        # Add this step's portfolio return to the returns history for Sharpe ratio calculation
        self.returns_history.append(portfolio_return)

        # Update the peak value for drawdown calculation
        self.peak_value = max(self.peak_value, self.cumulative_return)

        # Calculate drawdown
        drawdown = (self.peak_value - self.cumulative_return) / self.peak_value if self.peak_value > 0 else 0

        # Increment the current step
        self.current_step += 1
        done = self.current_step >= self.n_steps  # Check if the episode is over

        # Calculate reward based on Sharpe ratio adjusted for drawdown
        if len(self.returns_history) > 1:
            mean_return = np.mean(self.returns_history)
            volatility = np.std(self.returns_history)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            reward = sharpe_ratio - drawdown  # Penalize drawdown
        else:
            reward = portfolio_return  # Use portfolio return as the reward for the first step

        # Define termination and truncation flags
        terminated = done
        truncated = False

        # Get the next state (next simple returns)
        next_state = np.concatenate(([weights.sum()], self.simple_returns[self.current_step])).astype(
            np.float32) if not done else None

        # Return the next state, reward, and info (including portfolio return for PPO tracking)
        info = {'portfolio_return': portfolio_return, 'cumulative_return': self.cumulative_return}
        return next_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Portfolio Weights: {self.weights}, Cumulative Return: {self.cumulative_return}, Drawdown: {self.peak_value - self.cumulative_return}")
