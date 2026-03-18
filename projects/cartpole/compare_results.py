import numpy as np
import matplotlib.pyplot as plt
import os
import glob

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

def plot_comparison():
    # Find latest runs
    reinforce_dir = find_latest_run("run") # REINFORCE dirs start with "run_"
    dqn_dir = find_latest_run("dqn_run")   # DQN dirs start with "dqn_run_"
    
    if not reinforce_dir or not dqn_dir:
        print("Could not find results for both algorithms.")
        return

    print(f"Comparing:\nREINFORCE: {reinforce_dir}\nDQN: {dqn_dir}")

    # Load rewards
    r_rewards = np.load(os.path.join(reinforce_dir, "rewards.npy"))
    d_rewards = np.load(os.path.join(dqn_dir, "rewards.npy"))

    # Moving average
    def moving_average(data, window=50):
        if len(data) < window:
            return None
        return np.convolve(data, np.ones(window), 'valid') / window

    plt.figure(figsize=(10, 6))
    
    # Plot raw data (faint)
    plt.plot(r_rewards, alpha=0.3, color='blue', label='REINFORCE (Raw)')
    plt.plot(d_rewards, alpha=0.3, color='orange', label='DQN (Raw)')
    
    # Plot smoothed data (bold)
    r_smooth = moving_average(r_rewards)
    d_smooth = moving_average(d_rewards)
    if r_smooth is not None:
        plt.plot(r_smooth, color='blue', linewidth=2, label='REINFORCE (Smoothed)')
    if d_smooth is not None:
        plt.plot(d_smooth, color='orange', linewidth=2, label='DQN (Smoothed)')

    plt.title("Learning Curve Comparison: REINFORCE vs DQN")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, "comparison.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_comparison()
