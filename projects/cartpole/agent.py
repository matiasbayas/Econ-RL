import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import collections
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, n_actions))
                                
    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


def collect_trajectory(env, policy, gamma=0.99, render=False, device='cpu'):
    """
    Collects a trajectory by running the policy in the environment.
    Args:
        env: The environment to run the policy in.
        policy: The policy to use for action selection.
        gamma: The discount factor.
        render: Whether to render the environment.
        device: The device to run the policy on.
    Returns:
        Tuple[List[Categorical], List[float]]: The log probabilities of the actions taken and the rewards received.
    """
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        if render:
            env.render()
        # Convert observation to tensor on the correct device
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        # Get action from (stochastic) policy
        dist = policy(state_t)
        action = dist.sample()
        # Get log probability of action
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        # Take action in environment and update state
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        # Check if episode is done
        done = terminated or truncated
        
    return log_probs, rewards


def compute_returns(rewards, gamma=0.99):
    """
    Computes reward-to-go for a given list of rewards.
    Args:
        rewards: List of rewards.
        gamma: Discount factor.
    Returns:
        torch.Tensor: Returns for each time step.
    """
    G = 0.0
    returns = collections.deque()
    for r in reversed(rewards):
        G = r + gamma * G
        returns.appendleft(G)
    returns = torch.tensor(list(returns), dtype=torch.float32)
    # returns = (returns - returns.mean()) / (returns.std() + 1E-8) 
    return returns

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(state_dim, hidden_dim=64, n_actions=n_actions)
    log_probs, rewards = collect_trajectory(env, policy, gamma=0.99, render=False)
    print("Episode length:", len(rewards))
    print("Episode reward:", sum(rewards))


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(state_dim, hidden_dim=64, n_actions=n_actions)
    log_probs, rewards = collect_trajectory(env, policy, gamma=0.99, render=True)
    
    returns = compute_returns(rewards)
    print("First few rewards:", rewards[:5])
    print("First few returns:", returns[:5])
