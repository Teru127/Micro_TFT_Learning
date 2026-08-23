import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curve():
    log_file = "./logs/monitor.csv"

    if not os.path.exists(log_file):
        print(f"Monitor file not found at {log_file}:")
        return

    df = pd.read_csv(log_file, skiprows=1)
    print("Successfully read monitor.csv, data columns:", df.columns.tolist())

    rewards = df['r']

    timesteps = df['l'].cumsum()

    plt.figure(figsize=(12, 6))

    plt.plot(timesteps, rewards, label='Reward per Episode', color='blue', alpha = 0.3)

    window_size = 10
    smoothed_rewards = rewards.rolling(window=window_size).mean()
    plt.plot(timesteps, smoothed_rewards, label=f'Smoothed Reward (window={window_size})', color='red', linewidth=2)

    plt.axhline(y = 21.65, color = 'green', linestyle = '--', label = 'Random Baseline (21.65)')
    plt.axhline(y = 500, color = 'orange', linestyle = '--', label = 'Max Limit (500)')

    plt.title('Training Curve of PPO on CartPole-v1', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Total Timesteps', fontsize = 12)
    plt.ylabel('Episode Reward', fontsize = 12)
    plt.legend()    
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_curve()






