import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def run_ppo_with_monitor():
    print("Running PPO with monitoring on CartPole-v1...")
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    print("Creating environment...")
    env = gym.make("CartPole-v1")
    env = Monitor(env, log_dir)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Starting training 100,000 timesteps...")
    model.learn(total_timesteps=100000)
    print("Training completed.")

    model_path = os.path.join(os.path.dirname(__file__), "ppo_cartpole_model")
    model.save(model_path)
    print(f"Model saved to: {model_path}.zip")

    env.close()

    monitor_file = os.path.join(log_dir, "monitor.csv")
    if os.path.exists(monitor_file):
        print(f"Monitor file created at: {monitor_file}")

        with open(monitor_file, "r") as f:
            lines = f.readlines()
            print(f"Total lines in monitor file: {len(lines)}")
            print("First 5 lines of the monitor file:")
            for line in lines[:5]:
                print(line.strip())
    else:
        print("Monitor file not found.")

if __name__ == "__main__":
    run_ppo_with_monitor()