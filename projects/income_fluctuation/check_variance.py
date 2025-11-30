import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os
# Add the directory containing this script to the path
sys.path.append(os.path.dirname(__file__))

from agent import PolicyNet
from env import IncomeFluctuationEnv

def check_variance(run_dirs):
    env = IncomeFluctuationEnv()
    device = torch.device("cpu")
    
    # Define a grid of states to evaluate
    assets = np.linspace(env.amin, 5.0, 100)
    income_indices = [0, 1]
    
    print(f"{'Run Directory':<40} | {'Avg Std Dev':<15} | {'Min Std':<10} | {'Max Std':<10}")
    print("-" * 85)
    
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        model_path = run_path / "model.pth"
        
        if not model_path.exists():
            print(f"Skipping {run_dir} (no model.pth)")
            continue
            
        # Load Policy
        policy = PolicyNet(env.observation_space.shape[0]).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
        
        stds = []
        with torch.no_grad():
            for a in assets:
                for inc_idx in income_indices:
                    state = np.array([a, inc_idx], dtype=np.float32)
                    state_t = torch.tensor(state).unsqueeze(0).to(device)
                    _, std = policy(state_t)
                    stds.append(std.item())
        
        avg_std = np.mean(stds)
        min_std = np.min(stds)
        max_std = np.max(stds)
        
        print(f"{run_path.name:<40} | {avg_std:<15.6f} | {min_std:<10.6f} | {max_std:<10.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="List of run directories")
    args = parser.parse_args()
    
    check_variance(args.run_dirs)
