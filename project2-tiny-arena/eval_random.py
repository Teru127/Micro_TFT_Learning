import numpy as np
from env import TinyArenaEnv

def eval_random(n=200, seed=0):
    env = TinyArenaEnv()
    rewards, wins = [], 0
    for i in range(n):
        obs, _ = env.reset(seed=seed+i)
        total = 0
        win = False
        done = False
        while not done:
            obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
            total += rew
            done = terminated or truncated
            if terminated and env.state.agent.is_alive:
                win = True
        rewards.append(total)
        if win:
            wins += 1
    print(f"random baseline over {n} episodes:")
    print(f"mean {np.mean(rewards):.2f}, std {np.std(rewards):.2f}")
    print(f"win rate {wins/n:.1%}")

if __name__ == "__main__":
    eval_random(200)
