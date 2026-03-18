import numpy as np

from env import IncomeFluctuationEnv

def test_env():
    print("Initializing environment...")
    env = IncomeFluctuationEnv()
    
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial Observation: {obs}")
    assert len(obs) == 2
    assert obs[0] >= env.amin
    
    print("Taking a step (savings_rate=0.5)...")
    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Next Observation: {next_obs}")
    print(f"Reward: {reward}")
    
    assert len(next_obs) == 2
    assert next_obs[0] >= 0
    
    print("\nRunning 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward.item():.4f}")
        
    print("\nSUCCESS: Environment passed basic checks.")

if __name__ == "__main__":
    test_env()
