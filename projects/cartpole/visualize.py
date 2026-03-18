import matplotlib.pyplot as plt
import torch
import gymnasium as gym
import numpy as np
import os
from agent import PolicyNet, collect_trajectory

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    
    # Calculate moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Avg', color='orange')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()

def watch_agent(model_path=None):
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = PolicyNet(obs_dim, hidden_dim=64, n_actions=n_actions)
    
    if model_path and os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print(f"Loaded model from {model_path}")
    else:
        print("Using random weights (Untrained Agent)")
        
    print("Running 5 episodes...")
    for _ in range(5):
        _, rewards = collect_trajectory(env, policy, render=True)
        print(f"Reward: {sum(rewards)}")
    
    env.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualize.py [plot|watch] [run_dir]")
        print("Example: python visualize.py plot projects/cartpole/results/run_2023-10-27_14-00-00")
        sys.exit(1)
        
    command = sys.argv[1]
    run_dir = sys.argv[2]
    
    if command == "plot":
        rewards_path = os.path.join(run_dir, "rewards.npy")
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path)
            plot_path = os.path.join(run_dir, "training_curve.png")
            plot_rewards(rewards, save_path=plot_path)
        else:
            print(f"Error: {rewards_path} not found.")
            
    elif command == "watch":
        model_path = os.path.join(run_dir, "model.pth")
        watch_agent(model_path)
        
    else:
        print(f"Unknown command: {command}")
