import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import IncomeFlucuationEnv
from agent import PolicyNet, collect_trajectory, compute_returns

def train_reinforce(n_episodes=1000, lr=1E-3, seed=42, device='cpu', use_discounted_gradient=True):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = IncomeFlucuationEnv(beta=0.96, sigma=2.0, y=[0.5, 1.5], P=[[0.1, 0.9], [0.9, 0.1]], amin=0.0)
    
    obs_dim = env.observation_space.shape[0]

    policy = PolicyNet(obs_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    pbar = tqdm(range(n_episodes))
    episode_rewards = []

    for i in pbar:
        # 1. Collect trajectory
        log_probs, rewards = collect_trajectory(env, policy, max_steps=100, device=device)
        returns = compute_returns(rewards, gamma=env.beta)  

        # 2. Compute loss
        log_probs_t = torch.stack(log_probs) 
        if use_discounted_gradient:
            # Create discount factors efficiently [1, gamma, gamma^2, ..., gamma^(T-1)]
            T = len(rewards)
            discounts = torch.logspace(0, T-1, steps=T, base=env.beta, device=device)
            loss = - (discounts * log_probs_t * returns).sum()
        else:
            loss = - (log_probs_t * returns).sum()
        
        # 3. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Logging
        discounted_total_reward = returns[0].item()
        episode_rewards.append(discounted_total_reward)
        pbar.set_postfix(discounted_total_reward=discounted_total_reward)

    return policy, episode_rewards


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Income Fluctuation Agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--discounted_gradient", action="store_true", help="Use discounted gradient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    policy, rewards = train_reinforce(
        n_episodes=args.episodes, 
        lr=args.lr, 
        seed=args.seed, 
        use_discounted_gradient=args.discounted_gradient
    )
    
    # Plot learning curve
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Discounted Reward")
    plt.title(f"Income Fluctuation (Discounted Grad: {args.discounted_gradient})")
    plt.savefig("projects/income_fluctuation/learning_curve.png")
    print("Training complete! Saved plot.")