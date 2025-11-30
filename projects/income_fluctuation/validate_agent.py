import torch
import numpy as np
import matplotlib.pyplot as plt
from env import IncomeFluctuationEnv
from agent import PolicyNet
import argparse
from pathlib import Path

def validate(run_dir):
    run_path = Path(run_dir)
    model_path = run_path / "model.pth"
    
    # Load environment and model
    env = IncomeFluctuationEnv(beta=0.96, sigma=2.0, amin=0.0)
    obs_dim = env.observation_space.shape[0]
    
    policy = PolicyNet(obs_dim, hidden_dim=64)
    try:
        policy.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    policy.eval()
    
    # Define asset grid for plotting
    assets = np.linspace(env.amin, 5.0, 100)
    
    # Arrays to store consumption and next assets
    c_low, c_high = [], []
    ap_low, ap_high = [], []
    
    with torch.no_grad():
        for a in assets:
            # Low Income (Index 0)
            state_low = np.array([a, 0.0], dtype=np.float32)
            state_t_low = torch.tensor(state_low).unsqueeze(0)
            mean_low, _ = policy(state_t_low)
            s_rate_low = torch.sigmoid(mean_low).item()
            
            res_low = (1 + env.r) * a + env.y[0]
            cons_low = res_low * (1 - s_rate_low)
            ap_low_val = res_low * s_rate_low
            
            c_low.append(cons_low)
            ap_low.append(ap_low_val)
            
            # High Income (Index 1)
            state_high = np.array([a, 1.0], dtype=np.float32)
            state_t_high = torch.tensor(state_high).unsqueeze(0)
            mean_high, _ = policy(state_t_high)
            s_rate_high = torch.sigmoid(mean_high).item()
            
            res_high = (1 + env.r) * a + env.y[1]
            cons_high = res_high * (1 - s_rate_high)
            ap_high_val = res_high * s_rate_high
            
            c_high.append(cons_high)
            ap_high.append(ap_high_val)
            
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Consumption
    ax1.plot(assets, c_low, label=f"Low Income (y={env.y[0]})", color='red')
    ax1.plot(assets, c_high, label=f"High Income (y={env.y[1]})", color='blue')
    ax1.plot(assets, assets * (env.r), 'k--', alpha=0.3, label="Interest Income (r*a)") # Reference
    ax1.set_xlabel("Assets (a)")
    ax1.set_ylabel("Consumption (c)")
    ax1.set_title("Consumption Function c(a, y)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Next Assets (Savings)
    ax2.plot(assets, ap_low, label=f"Low Income (y={env.y[0]})", color='red')
    ax2.plot(assets, ap_high, label=f"High Income (y={env.y[1]})", color='blue')
    ax2.plot(assets, assets, 'k--', alpha=0.3, label="45 degree line (a'=a)") # Reference
    ax2.set_xlabel("Assets (a)")
    ax2.set_ylabel("Next Assets (a')")
    ax2.set_title("Asset Policy Function a'(a, y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = run_path / "policy_functions.png"
    plt.savefig(save_path)
    print(f"Saved policy plots to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Income Fluctuation Agent")
    parser.add_argument("run_dir", type=str, help="Path to the run directory (e.g., projects/income_fluctuation/results/run_...)")
    args = parser.parse_args()
    
    validate(args.run_dir)
