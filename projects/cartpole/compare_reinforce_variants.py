import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from train import train_reinforce

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def find_latest_run(prefix):
    # Find the latest run directory for the given algorithm
    search_path = os.path.join(RESULTS_DIR, f"{prefix}*")
    runs = [d for d in glob.glob(search_path) if os.path.isdir(d)]
    if not runs:
        print(f"No runs found for prefix: {prefix}")
        return None
    return max(runs, key=os.path.getctime)

def run_experiments():
    print("Running Standard REINFORCE...")
    train_reinforce(n_episodes=800, gamma=0.99, lr=1E-3, print_every=50, device_name="cpu", use_discounted_gradient=False)
    
    print("\nRunning Discounted Gradient REINFORCE...")
    train_reinforce(n_episodes=800, gamma=0.99, lr=1E-3, print_every=50, device_name="cpu", use_discounted_gradient=True)

def plot_comparison():
    # Find latest runs
    std_dir = find_latest_run("run_standard")
    disc_dir = find_latest_run("run_discounted")
    
    if not std_dir or not disc_dir:
        print("Could not find results for both variants.")
        return

    print(f"Comparing:\nStandard: {std_dir}\nDiscounted: {disc_dir}")

    # Load rewards
    std_rewards = np.load(os.path.join(std_dir, "rewards.npy"))
    disc_rewards = np.load(os.path.join(disc_dir, "rewards.npy"))

    # Moving average
    def moving_average(data, window=50):
        if len(data) < window:
            return None
        return np.convolve(data, np.ones(window), 'valid') / window

    plt.figure(figsize=(10, 6))
    
    # Plot raw data (faint)
    plt.plot(std_rewards, alpha=0.3, color='blue', label='Standard (Raw)')
    plt.plot(disc_rewards, alpha=0.3, color='red', label='Discounted Gradient (Raw)')
    
    # Plot smoothed data (bold)
    std_smooth = moving_average(std_rewards)
    disc_smooth = moving_average(disc_rewards)
    if std_smooth is not None:
        plt.plot(std_smooth, color='blue', linewidth=2, label='Standard (Smoothed)')
    if disc_smooth is not None:
        plt.plot(disc_smooth, color='red', linewidth=2, label='Discounted Gradient (Smoothed)')

    plt.title("REINFORCE: Standard vs Discounted Gradient")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, "reinforce_comparison.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    run_experiments()
    plot_comparison()
