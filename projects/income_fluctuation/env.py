import gymnasium as gym
import numpy as np
from gymnasium import spaces


class IncomeFluctuationEnv(gym.Env):
    def __init__(self, r=0.02, beta=0.96, sigma=2.0, y=[0.5, 1.5], P=[[0.1, 0.9], [0.9, 0.1]], amin=0.0):

        # Parameters
        self.r = r
        self.beta = beta
        self.sigma = sigma
        self.y = np.array(y)
        self.N_y = len(self.y)
        self.P = np.array(P)
        self.amin = amin

        assert self.P.shape == (self.N_y, self.N_y)

        # Action space: 
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: [Assets, Income_index] 
        # Assets are unbounded, Income_index is bounded between 0 and N_y-1
        self.observation_space = spaces.Box(
            low=np.array([self.amin, 0.0]),
            high=np.array([np.inf, self.N_y-1]),
            dtype=np.float32
        )

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly choose initial assets and income
        initial_assets = self.np_random.uniform(low=self.amin, high=self.amin + 2.0)
        initial_income_idx = self.np_random.integers(0, self.N_y)

        self.state = np.array([initial_assets, initial_income_idx], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        action = action.item() if hasattr(action, "item") else float(action)

        # 1. Unpack state
        a, income_idx = self.state
        income_idx = int(income_idx)

        # 2. Calculate total resources
        income = self.y[income_idx]
        resources = (1 + self.r) * a + income

        # 3. Process action (assumed to be a saving rate)
        savings_rate = float(np.clip(action, 0.0, 1.0))
        a_plus = resources * savings_rate
        c = resources - a_plus

        # 4. Reward 
        c = max(c, 1E-5)
        if self.sigma == 1.0:
            reward = np.log(c)
        else:
            reward = (c ** (1 - self.sigma)) / (1 - self.sigma)
        
        # 5. Update state
        next_income_probs = self.P[income_idx, :]
        next_income_idx = self.np_random.choice(self.N_y, p=next_income_probs)
        
        self.state = np.array([a_plus, next_income_idx], dtype=np.float32)

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}

        


