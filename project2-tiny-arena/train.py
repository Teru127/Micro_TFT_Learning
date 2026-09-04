import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env import TinyArenaEnv

env = Monitor(TinyArenaEnv(), filename="monitor")
model = PPO("MlpPolicy", env, verbose=1, seed=0)
model.learn(total_timesteps=100_000)
model.save("ppo_tiny_arena")
print("saved ppo_tiny_arena.zip")