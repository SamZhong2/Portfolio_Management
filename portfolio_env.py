import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(PortfolioEnv, self).__init__()

        # Assets in portfolio
        self.previous_portfolio_value = None
        self.current_portfolio = None
        self.current_step = None
        self.data = data
        self.initial_cash = 1000
        self.cash = self.initial_cash
        self.returns = []

        # Extract feature columns
        self.price_columns = [col for col in data.columns if '_Price' in col]
        self.num_assets = len(self.price_columns)
        self.feature_columns = [col for col in data.columns if col != 'Date']

        # Action space is reallocation of num_assets + 1 accounting for cash, and they add to 1.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32
        )

        # Not sure yet, include many things
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_columns),), dtype=np.float32
        )

        # Initialize environment state
        self.shares_held = np.zeros(self.num_assets)

    def reset(self, seed=None, options=None):
        # Reset environment
        self.current_step = 0
        self.current_portfolio = np.zeros(self.num_assets + 1)
        self.current_portfolio[-1] = 1.0  # Start fully in cash
        self.previous_portfolio_value = self.initial_cash
        return self._get_observation(), {}

    def _get_observation(self):
        # Get feature data for the current timestep
        current_features = self.data.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
        return current_features

    def _compute_reward(self):
        # # Get current prices
        # current_prices = self.data.iloc[self.current_step][self.price_columns].values
        #
        # # Calculate portfolio value: shares * prices + cash
        # portfolio_value = np.sum(self.shares_held * current_prices) + self.cash
        #
        # # Reward: Relative change in portfolio value
        # reward = (portfolio_value - self.previous_portfolio_value)
        #
        # self.previous_portfolio_value = portfolio_value

        # Get current prices
        current_prices = self.data.iloc[self.current_step][self.price_columns].values

        # Calculate portfolio value: shares * prices + cash
        portfolio_value = np.sum(self.shares_held * current_prices) + self.cash

        # Calculate return for the current step
        portfolio_return = (portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Update returns list
        self.returns.append(portfolio_return)

        # Calculate Sharpe Ratio
        risk_free_rate = 0  # Assuming a risk-free rate of 0 for simplicity
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            volatility = np.std(self.returns)
            sharpe_ratio = (mean_return - risk_free_rate) / (
                        volatility + 1e-8)  # Add small value to avoid division by zero
        else:
            sharpe_ratio = 0  # Not enough data to calculate Sharpe Ratio

        # Update previous portfolio value
        self.previous_portfolio_value = portfolio_value
        return sharpe_ratio

    def step(self, action):
        # Map actions from [-1, 1] to [0, 1]
        action = (action + 1) / 2  # Scale from [-1, 1] to [0, 1]

        # Normalize to ensure actions sum to 1
        action = np.clip(action, 0, 1)
        total = np.sum(action)
        if not np.isclose(total, 1.0):
            # Normalize action if it doesn't sum to 1 (for compatibility with check_env)
            action = action / total

        # Get current prices
        current_prices = self.data.iloc[self.current_step][self.price_columns].values

        # Update shares held based on the action
        target_allocation = action[:-1] * (self.cash + np.sum(self.shares_held * current_prices))
        self.cash = action[-1] * (self.cash + np.sum(self.shares_held * current_prices))

        for i in range(self.num_assets):
            self.shares_held[i] = target_allocation[i] / current_prices[i]

        # Advance to the next timestep
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # Define a truncation condition if applicable (e.g., max steps)

        # Compute reward
        reward = self._compute_reward()

        return self._get_observation(), reward, terminated, truncated, {}
