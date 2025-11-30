import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import IncomeFluctuationEnv
from agent import PolicyNet, collect_trajectory, compute_returns
import argparse
from pathlib import Path
from datetime import datetime

def train_reinforce(n_episodes=1000, lr=1E-3, seed=42, device='cpu', use_discounted_gradient=True, batch_size=10):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("projects/income_fluctuation/results") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {run_dir}")

    env = IncomeFluctuationEnv(beta=0.96, sigma=2.0, y=[0.5, 1.5], P=[[0.1, 0.9], [0.9, 0.1]], amin=0.0)
    
    obs_dim = env.observation_space.shape[0]

    policy = PolicyNet(obs_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    episode_rewards = []

    # Calculate how many batches we need
    n_batches = n_episodes // batch_size
    pbar = tqdm(range(n_batches))

    for b in pbar:

        batch_loss = 0
        batch_rewards = []

        for _ in range(batch_size):
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

            batch_loss += loss / batch_size
            batch_rewards.append(returns[0].item())
            
        # 3. Update (once per batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # 4. Logging
        # We extend the list with ALL rewards from this batch
        episode_rewards.extend(batch_rewards)
        
        # Update progress bar with rolling average of the last 50 episodes
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        pbar.set_description(f"Avg Reward: {avg_reward:.2f}")

    return policy, episode_rewards, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Income Fluctuation Agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--discounted_gradient", action="store_true", help="Use discounted gradient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    
    args = parser.parse_args()
    
    policy, rewards, run_dir = train_reinforce(
        n_episodes=args.episodes, 
        lr=args.lr, 
        seed=args.seed, 
        use_discounted_gradient=args.discounted_gradient,
        batch_size=args.batch_size
    )
    
    # Save model and rewards
    torch.save(policy.state_dict(), run_dir / "model.pth")
    np.save(run_dir / "rewards.npy", np.array(rewards))
    
    # Plot learning curve with smoothing
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Raw Reward', alpha=0.3)
    
    # Calculate moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Avg', color='orange')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Discounted Reward")
    plt.title(f"Income Fluctuation (Discounted Grad: {args.discounted_gradient})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / "learning_curve.png")
    print(f"Training complete! Results saved to {run_dir}")