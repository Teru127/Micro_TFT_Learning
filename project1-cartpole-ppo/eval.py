import time
import os
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    model_path = os.path.join(os.path.dirname(__file__), "ppo_cartpole_model")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.zip. "
            "Run cartpole_ppo.py once to save the PPO model."
        )

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)
    
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(20):
        observation, info = env.reset()
        total_reward = 0
        
        print(f"--- Loading episode {episode + 1} ---")
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            time.sleep(0.02)
            
            if terminated or truncated:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                break
                
    env.close()
    print("Finished evaluating!")

if __name__ == "__main__":
    main()