try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    # Fall back to gym if gymnasium isn't available
    import gym
    from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df_path, window_size=30, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = pd.read_csv(df_path)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_12']

        # Clean and coerce feature columns to numeric. Some CSVs may contain non-numeric rows (e.g. leftover labels).
        missing_cols = set(self.feature_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required feature columns: {missing_cols}")
        for c in self.feature_columns:
            # convert and coerce invalid strings to NaN
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        # drop rows where any feature is NaN (these were non-numeric or incomplete)
        self.df = self.df.dropna(subset=self.feature_columns).reset_index(drop=True)
        if len(self.df) <= self.window_size:
            raise ValueError(f"Dataframe too short after cleaning: {len(self.df)} rows (need > window_size={self.window_size})")
        
        # Actions: 0=Hold, 1=Buy (all-in), 2=Sell (all-out)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, len(self.feature_columns)), dtype=np.float32)
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        return self._get_observation()
    
    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        return self.df[self.feature_columns].iloc[start:end].values.astype(np.float32)
    
    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth
        self.current_step += 1
        
        if action == 1:  # Buy
            if self.balance > 0:
                self.shares_held += self.balance / current_price
                self.balance = 0
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        
        # Net worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Reward: return - drawdown penalty
        reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        reward -= 0.01 * drawdown
        
        done = self.current_step >= len(self.df) - 1 or self.net_worth <= 0
        return self._get_observation(), reward, done, {}
    
    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}, Net Worth: {self.net_worth:.2f}")\
    