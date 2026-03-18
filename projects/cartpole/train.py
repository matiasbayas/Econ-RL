import os
# Set fallback for MPS before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from agent import PolicyNet, collect_trajectory, compute_returns

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = SCRIPT_DIR / "results"

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_reinforce(
    n_episodes: int = 500, 
    gamma: float = 0.99, 
    lr: float = 1E-3, 
    print_every: int = 10, 
    device_name: str = "auto", 
    use_discounted_gradient: bool = False,
    seed: int = 42
) -> List[float]:
    """
    Trains the agent using simple REINFORCE algorithm.

    Args:
        n_episodes (int): Number of episodes to train.
        gamma (float): Discount factor.
        lr (float): Learning rate.
        print_every (int): Frequency of printing progress (deprecated by tqdm).
        device_name (str): Device to use ('auto', 'cpu', 'cuda', 'mps').
        use_discounted_gradient (bool): Whether to use the discounted gradient term.
        seed (int): Random seed for reproducibility.

    Returns:
        List[float]: List of total rewards per episode.
    """
    
    # Set seed for reproducibility
    set_seed(seed)

    # Create results directory using pathlib
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    variant = "discounted" if use_discounted_gradient else "standard"
    run_dir = RESULTS_DIR / f"run_{variant}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {run_dir}")

    # Device configuration
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Use context manager for environment
    with gym.make("CartPole-v1") as env:
        # Seed the environment
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        
        policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        episode_rewards = []
        
        # Use tqdm for progress bar
        pbar = tqdm(range(n_episodes), desc="Training")
        
        for ep in pbar:

            # Rollout trajectory
            log_probs, rewards = collect_trajectory(env, policy, gamma=gamma, device=device)
            returns = compute_returns(rewards, gamma=gamma).to(device)

            # Stack log probs into tensor 
            log_probs_t = torch.stack(log_probs) 
            
            # Compute loss  
            if use_discounted_gradient:
                # Create discount factors efficiently [1, gamma, gamma^2, ..., gamma^(T-1)]
                T = len(rewards)
                discounts = torch.logspace(0, T-1, steps=T, base=gamma, device=device)
                loss = - (discounts * log_probs_t * returns).sum()
            else:
                loss = - (log_probs_t * returns).sum()
            # Clear gradients
            optimizer.zero_grad() 
            # Backpropagate
            loss.backward() 
            # Update weights
            optimizer.step() 
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            
            # Update progress bar description with moving average
            avg_last = np.mean(episode_rewards[-print_every:]) if len(episode_rewards) >= print_every else np.mean(episode_rewards)
            pbar.set_description(f"Last: {total_reward:.0f} | Avg: {avg_last:.1f}")
        
        # Save the trained model
        torch.save(policy.state_dict(), run_dir / "model.pth")
        print(f"Model saved to {run_dir / 'model.pth'}")
        
        # Save rewards for plotting
        np.save(run_dir / "rewards.npy", np.array(episode_rewards))
        print(f"Rewards saved to {run_dir / 'rewards.npy'}")
        
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CartPole Agent")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use for training")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=5E-4, help="Learning rate")
    parser.add_argument("--discounted_gradient", action="store_true", help="Use discounted gradient (gamma^t)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    rewards = train_reinforce(
        n_episodes=args.episodes, 
        gamma=0.99, 
        lr=args.lr, 
        print_every=10, 
        device_name=args.device, 
        use_discounted_gradient=args.discounted_gradient,
        seed=args.seed
    )
