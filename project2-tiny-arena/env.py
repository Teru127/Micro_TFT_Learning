import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from models import Unit, BattleState
from battle_turn import take_turn

class TinyArenaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0,0,15,15,0,0], dtype=np.float32),
            high=np.array([100,100,25,25,50,50], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: attack, 1: defend

    def _get_obs(self):
        return np.array(self.state.to_observation(), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.state = BattleState(
            agent=Unit(100, rng.uniform(15, 25), rng.uniform(0, 50)),
            opp=Unit(100, rng.uniform(15, 25), rng.uniform(0, 50))
        )
        self.turn = 0
        return self._get_obs(), {}

    def step(self, action):
        opp_action = int(self.np_random.integers(0, 2))
        self.state = take_turn(self.state, int(action), opp_action)
        self.turn += 1
        terminated = not self.state.agent.is_alive or not self.state.opp.is_alive
        truncated = self.turn >= 50 and not terminated
        return self._get_obs(), 0.0, terminated, truncated, {}
    
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    check_env(TinyArenaEnv(), warn=True)
    print("check_env passed")
    env = TinyArenaEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    print("100 steps ok")