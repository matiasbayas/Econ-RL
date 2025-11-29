import gymnasium as gym
import torch
import numpy as np
import os
import argparse
from datetime import datetime
from agent import PolicyNet, collect_trajectory, compute_returns

def train_reinforce(n_episodes=500, gamma=0.99, lr=1E-3, print_every=10, device_name="auto"):
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("projects", "cartpole", "results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to {run_dir}")

    # Device configuration
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Set fallback for MPS if needed
    if device.type == 'mps':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    episode_rewards = []
    
    for ep in range(n_episodes):
        log_probs, rewards = collect_trajectory(env, policy, gamma=gamma, device=device)
        returns = compute_returns(rewards, gamma=gamma).to(device)

        # stack log probs into tensor 
        log_probs_t = torch.stack(log_probs) 
        loss = - (log_probs_t * returns).sum()

        optimizer.zero_grad() # clear gradients
        loss.backward() # backpropagate
        optimizer.step() # update weights
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        if ep % print_every == 0:
            avg_last = np.mean(episode_rewards[-print_every:])
            print(f"Episode {ep+1}:, Last Reward = {total_reward}, Avg Reward = {avg_last}")
    
    # Save the trained model
    torch.save(policy.state_dict(), os.path.join(run_dir, "model.pth"))
    print(f"Model saved to {run_dir}/model.pth")
    
    # Save rewards for plotting
    np.save(os.path.join(run_dir, "rewards.npy"), np.array(episode_rewards))
    print(f"Rewards saved to {run_dir}/rewards.npy")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CartPole Agent")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use for training")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=5E-4, help="Learning rate")
    
    args = parser.parse_args()
    
    rewards = train_reinforce(n_episodes=args.episodes, gamma=0.99, lr=args.lr, print_every=10, device_name=args.device)
