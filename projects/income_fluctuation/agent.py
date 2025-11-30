import torch 
import collections
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Two heads: one for Mean, one for LogStd
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        std = torch.exp(log_std)

        return mean, std

    def get_action(self, state, device):
        # 1. Forward Pass
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mean, std = self.forward(state_t)
        
        # 2. Create Distribution and Sample Raw Action
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()

        # 3. Apply Sigmoid to Get Action in [0, 1]
        action = torch.sigmoid(raw_action)

        # 4. Compute Log Prob with Jacobian correction (change of variables)
        # log_prob(a) = log_prob(x) - log(f'(x))
        # f'(x) = sigmoid(x) * (1 - sigmoid(x)) = action * (1 - action)
        log_prob = dist.log_prob(raw_action) - torch.log(action * (1 - action) + 1E-6)
        log_prob = log_prob.sum(dim=-1)

        return action.item(), log_prob



def collect_trajectory(env, policy, max_steps=100, device='cpu'):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    for _ in range(max_steps):

        action, log_prob = policy.get_action(state, device)
        next_state, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        if terminated or truncated:
            break

        state = next_state

    return log_probs, rewards
        

def compute_returns(rewards, gamma=0.99):
    G = 0.0
    returns = collections.deque()
    for r in reversed(rewards):
        G = r + gamma * G
        returns.appendleft(G)
    return torch.tensor(list(returns), dtype=torch.float32)