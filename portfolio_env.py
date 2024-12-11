import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(PortfolioEnv, self).__init__()

        self.data = data

        # Get the amount of assets
        self.price_columns = [col for col in data.columns if '_Price' in col]
        self.num_assets = len(self.price_columns)

        # Initialize the window size for the past 30 days
        self.window_size = int(252/4)
        
        # Action space is continuous with possible allocation from 0 to 1 for each asset and cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets + 1,))

        # Oberservatoin space is the everything except the raw price of the stock for the past 30 days
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size * (len(self.data.columns) - self.num_assets - 1),), dtype=np.float64)

        # Initialize the state of the portfolio
        self.portfolio = np.zeros(self.num_assets + 1)

        # Initialize the risk-free rate for sharpe ratio calculation
        self.risk_free_rate = 0.01 / 252

        # Initialize the horizon for the episode
        self.horizon = 252
        
        # Initialize the current step
        self.current_step = 0

        # Initialize the daily returns for the entire horizon
        self.daily_returns = []

        # # Initialize the portfolio value
        # self.portfolio_value = 1000

        # Initialize the maximum number of steps
        self.max_steps = 500
        
    # Gets the random day we are training for and the data for the past 30 days
    def _get_random_window(self):
        # 1. Randomly select a start index ensuring enough data for the window and horizon
        start_idx = np.random.randint(self.window_size, len(self.data) - self.horizon)
        
        # 2. Exclude the first column (date) and any price columns
        all_columns = self.data.columns[1:]  # Exclude the first column
        non_price_features = [col for col in all_columns if not col.endswith('_Price')]
        
        observation = self.data[non_price_features].iloc[start_idx - self.window_size : start_idx].values

        # Normalize observations (min-max scaling)
        obs_min = np.min(observation, axis=0)
        obs_max = np.max(observation, axis=0)
        normalized_obs = (observation - obs_min) / (obs_max - obs_min + 1e-8)

        # Flatten the observation for input to the agent
        return normalized_obs.flatten(), start_idx

    def step(self, action):
        # 1. Get the next observation window
        observation, start_idx = self._get_random_window()  # Random 30-day window ending at `start_idx`

        # 2. Parse and normalize the action (portfolio allocation)
        action = (action + 1) / 2  # Rescale to [0, 1]
        action = np.clip(action, 0, 1)  # Ensure valid weights
        action_sum = np.sum(action) + 1e-8  # Avoid division by zero
        action = action / action_sum  # Normalize to sum to 1

        asset_weights = action[:-1]  # First `self.num_assets` elements
        cash_weight = action[-1]    # Last element for cash allocation


        # 3. Simulate returns over the next 252 days
        end_idx = min(start_idx + self.horizon, len(self.data))  # Ensure we don't exceed dataset length
        return_columns = [col for col in self.data.columns if col.endswith('_Return')]
        future_returns = self.data[return_columns].iloc[start_idx:end_idx].values
        
        # Weighted sum of future returns for assets
        asset_returns = np.dot(future_returns, asset_weights)

        # Daily portfolio returns, including cash
        portfolio_returns = asset_returns + (cash_weight * self.risk_free_rate)

        # 4. Compute reward metrics
        # Cumulative return
        cumulative_return = np.prod(1 + portfolio_returns) - 1

        # Sharpe Ratio (annualized)
        mean_daily_return = np.mean(portfolio_returns)
        mean_return_annualized = (1 + mean_daily_return)**252 - 1
        volatility_annualized = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio_annualized = (mean_return_annualized - self.risk_free_rate * 252) / (volatility_annualized + 1e-8)

        # Sortino Ratio (annualized)
        # Filter for negative returns
        downside_returns = portfolio_returns[portfolio_returns < 0]

        # Compute downside deviation (annualized)
        downside_deviation = np.std(downside_returns) * np.sqrt(252)  # Scale for annualization

        # Compute annualized return
        mean_daily_return = np.mean(portfolio_returns)
        mean_return_annualized = (1 + mean_daily_return)**252 - 1

        # Sortino Ratio
        if downside_deviation > 0:
            sortino_ratio = (mean_return_annualized - self.risk_free_rate * 252) / downside_deviation
        else:
            sortino_ratio = 0  # Handle case with no downside deviation

        # Maximum Drawdown
        portfolio_values = np.cumprod(1 + portfolio_returns)
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)

        # Conditional Value at Risk (CVaR)
        confidence_level = 0.05
        sorted_returns = np.sort(portfolio_returns)
        var = np.percentile(sorted_returns, confidence_level * 100)  # Value at Risk
        cvar = sorted_returns[sorted_returns <= var].mean()  # CVaR

        # Reward: Combine metrics
        reward = (
            cumulative_return                  # Reward total returns
            + sharpe_ratio_annualized      # Reward overall risk-adjusted returns
            + 0.5 * sortino_ratio                # Reward downside risk-adjusted returns
            - max_drawdown                 # Penalize large drawdowns
            - abs(cvar)                        # Penalize extreme losses (tail risk)
        )
        
        # 5. Increment the step counter and check if the episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_steps  # True if max steps reached
        truncated = False  # Define conditions for truncation if applicable (e.g., timeout)

        # 6. Return observation, reward, done, and additional info
        return observation, reward, terminated, truncated, {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio_annualized,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'cvar': cvar
        }



    def reset(self, seed=None, options=None):
        self.current_step = 0
        # self.portfolio_value = 1000
        observation, _ = self._get_random_window()  # Random initial window
        return observation, {}


       


   