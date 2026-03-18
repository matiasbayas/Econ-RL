import gymnasium as gym

def test_env():
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = env.reset()
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    for _ in range(100):
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    print("Environment test passed!")

if __name__ == "__main__":
    test_env()
