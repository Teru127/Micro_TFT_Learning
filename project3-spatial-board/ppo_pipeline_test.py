"""Short PPO pipeline check against a Monte Carlo position-value baseline."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BOT_ATTACK_RANGE, BOT_START, PLAYER_ATTACK_RANGE, TinyArenaEnv


def estimate_position_values(games_per_position, seed):
    """Estimate each cell's average reward using the same seeds for every cell."""
    env = TinyArenaEnv()
    rewards = np.zeros(len(env.player_positions), dtype=np.float64)

    for action in range(env.action_space.n):
        total_reward = 0.0
        for game_seed in range(seed, seed + games_per_position):
            env.reset(seed=game_seed)
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        rewards[action] = total_reward / games_per_position

    return env.player_positions, rewards


def evaluate_deterministic_policy(model, games, seed):
    """Evaluate the final policy action over fresh randomized battles."""
    env = TinyArenaEnv()
    outcomes = {"A": 0, "B": 0, "draw": 0}
    total_reward = 0.0
    selected_position = None

    for game_seed in range(seed, seed + games):
        observation, _ = env.reset(seed=game_seed)
        action, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
        selected_position = env.player_positions[action]

        _, reward, _, _, info = env.step(action)
        total_reward += reward
        outcomes[info["outcome"]] += 1

    return selected_position, total_reward / games, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--oracle-games", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "ppo_pipeline_output",
    )
    args = parser.parse_args()

    if args.timesteps < 1 or args.oracle_games < 1 or args.eval_games < 1:
        parser.error("--timesteps, --oracle-games, and --eval-games must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = args.output_dir / "monitor.csv"
    curve_file = args.output_dir / "training_curve.png"
    oracle_file = args.output_dir / "position_oracle.csv"

    # Check the unwrapped environment against Gymnasium's expected API.
    check_env(TinyArenaEnv(), skip_render_check=True)

    # Monitor records episode rewards; DummyVecEnv supplies SB3's vector API.
    monitored_env = Monitor(TinyArenaEnv(), filename=str(monitor_file))
    vector_env = DummyVecEnv([lambda: monitored_env])
    model = PPO(
        "MlpPolicy",
        vector_env,
        n_steps=1024,
        batch_size=256,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps)

    # Training Monitor rewards can lag the final policy update; evaluate final weights
    # separately by letting the deterministic policy play fresh seeded battles.
    eval_seed = args.seed + args.oracle_games
    final_position, final_reward, final_outcomes = evaluate_deterministic_policy(
        model, args.eval_games, eval_seed
    )

    # Read the Monitor log and draw a rolling average to make the trend clearer.
    monitor_results = load_results(str(args.output_dir))
    episode_rewards = monitor_results["r"]
    rolling_rewards = episode_rewards.rolling(window=100, min_periods=1).mean()
    plt.plot(np.arange(1, len(rolling_rewards) + 1), rolling_rewards)
    plt.xlabel("Completed episodes")
    plt.ylabel("Training reward (rolling 100 episodes)")
    plt.title("PPO training log and final deterministic evaluation")
    plt.axhline(
        final_reward,
        color="red",
        linestyle="--",
        label=f"Final deterministic policy ({final_reward:.3f})",
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_file)
    plt.close()

    # Estimate how each action performs independently of PPO's policy.
    positions, oracle_values = estimate_position_values(args.oracle_games, args.seed)
    with oracle_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("action", "row", "column", "average_reward"))
        for action, (position, value) in enumerate(zip(positions, oracle_values)):
            writer.writerow((action, position[0], position[1], f"{value:.6f}"))

    # Compare the learned action probabilities with the empirical position ranking.
    observation, _ = TinyArenaEnv().reset(seed=args.seed)
    observation_tensor = torch.as_tensor(observation).unsqueeze(0)
    action_probabilities = (
        model.policy.get_distribution(observation_tensor)
        .distribution.probs.detach()
        .cpu()
        .numpy()[0]
    )
    oracle_order = np.argsort(-oracle_values)
    policy_order = np.argsort(-action_probabilities)

    print(f"Requested PPO timesteps: {args.timesteps}")
    print(f"Completed Monitor episodes: {len(episode_rewards)}")
    print(
        f"Last logged 100-episode training average (not final eval): "
        f"{rolling_rewards.iloc[-1]:.3f}"
    )
    print(
        f"Final deterministic evaluation over {args.eval_games} games: "
        f"position={final_position}, average reward={final_reward:.3f}, "
        f"wins={final_outcomes['A']}, draws={final_outcomes['draw']}, "
        f"losses={final_outcomes['B']}"
    )
    print(f"Bot start: {BOT_START}; ranges: A={PLAYER_ATTACK_RANGE}, B={BOT_ATTACK_RANGE}")
    print("Oracle ranking (position: average reward):")
    for action in oracle_order:
        print(
            f"  {positions[action]}: {oracle_values[action]:.3f} "
            f"(PPO probability {action_probabilities[action]:.3f})"
        )
    print("PPO action ranking:", [positions[action] for action in policy_order])
    print(f"Saved PPO reward curve to: {curve_file}")
    print(f"Saved position estimates to: {oracle_file}")
    print(f"Saved Monitor log to: {monitor_file}")

    vector_env.close()


if __name__ == "__main__":
    main()
