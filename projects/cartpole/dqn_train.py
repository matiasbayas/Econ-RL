import gymnasium as gym
import torch 
import numpy as np
import os
import argparse
from datetime import datetime
from dqn_agent import DQNAgent

def train_dqn(n_episodes=500, batch_size=64, device_name="auto", lr=1E-3):
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("projects", "cartpole", "results", f"dqn_run_{timestamp}")
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

    # Hyperparameters
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    # Setup
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, hidden_dim=64, n_actions=n_actions, lr=lr, gamma=0.99, buffer_capacity=10000, device=device)
    
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # 1. Act
            action = agent.select_action(state, epsilon)
            
            # 2. Observe
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. Remember
            agent.buffer.push(state, action, reward, next_state, done)
            
            # 4. Learn
            agent.update(batch_size)

            # 5. Update state
            state = next_state
            total_reward += reward
            
        # 6. Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)
        
        if ep % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep}: Reward {total_reward}, Avg Reward {avg_reward:.1f}, Epsilon {epsilon:.2f}")

    # Save results
    torch.save(agent.q_net.state_dict(), os.path.join(run_dir, "model.pth"))
    np.save(os.path.join(run_dir, "rewards.npy"), np.array(episode_rewards))
    print(f"Saved model and rewards to {run_dir}")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
    parser.add_argument("--lr", type=float, default=1E-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_dqn(n_episodes=args.episodes, device_name=args.device, lr=args.lr, batch_size=args.batch_size)

            
        
