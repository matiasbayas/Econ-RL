import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, n_actions))
                                
    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Performs a forward pass through the policy network.

        Args:
            obs (torch.Tensor): The observation tensor, with shape [batch_size, obs_dim].

        Returns:
            Categorical: A Categorical distribution over the possible actions.
        """
        logits = self.net(obs)
        return Categorical(logits=logits)


# Generate trajectory
def collect_trajectory(env, policy, gamma=0.99, render=False, device='cpu'):
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        if render:
            env.render()
        
        # Convert observation to tensor on the correct device
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        dist = policy(obs_t)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        
        done = terminated or truncated
        
    return log_probs, rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions)
    log_probs, rewards = collect_trajectory(env, policy, gamma=0.99, render=False)
    print("Episode length:", len(rewards))
    print("Episode reward:", sum(rewards))




def compute_returns(rewards, gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # returns = (returns - returns.mean()) / (returns.std() + 1E-8) 
    return returns

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions)
    log_probs, rewards = collect_trajectory(env, policy, gamma=0.99, render=True)
    
    returns = compute_returns(rewards)
    print("First few rewards:", rewards[:5])
    print("First few returns:", returns[:5])
