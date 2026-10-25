from collections import deque
import random

from env import (
    BOT_ATTACK_RANGE,
    BOT_START,
    PLAYER_ATTACK_RANGE,
    TinyArenaEnv,
)


def train_position_learner(
    total_games=5000,
    epsilon=0.15,
    player_attack_range=PLAYER_ATTACK_RANGE,
    bot_attack_range=BOT_ATTACK_RANGE,
):
    # Treat each legal starting cell as an action and learn its average reward.
    env = TinyArenaEnv(
        player_attack_range=player_attack_range,
        bot_attack_range=bot_attack_range,
    )
    positions = env.player_positions
    action_for_position = {
        position: action for action, position in enumerate(positions)
    }
    values = {position: 0.0 for position in positions}
    counts = {position: 0 for position in positions}
    records = {position: {"A": 0, "B": 0, "draw": 0} for position in positions}
    recent_rewards = deque(maxlen=100)
    learning_curve = []

    for _ in range(total_games):
        # Try every cell once before choosing the current best most of the time.
        untried = [position for position in positions if counts[position] == 0]
        if untried:
            player_position = random.choice(untried)
        elif random.random() < epsilon:
            player_position = random.choice(positions)
        else:
            best_value = max(values.values())
            best_positions = [
                position for position in positions if values[position] == best_value
            ]
            player_position = random.choice(best_positions)

        env.reset()
        _, _, _, _, info = env.step(action_for_position[player_position])
        outcome = info["outcome"]
        reward = {"A": 1.0, "draw": 0.5, "B": 0.0}[outcome]

        counts[player_position] += 1
        values[player_position] += (
            reward - values[player_position]
        ) / counts[player_position]
        records[player_position][outcome] += 1

        recent_rewards.append(reward)
        learning_curve.append(sum(recent_rewards) / len(recent_rewards))

    return values, counts, records, learning_curve


if __name__ == "__main__":
    # Train the learner against the bot and summarize performance by starting cell.
    import matplotlib.pyplot as plt

    total_games = 5000
    values, counts, records, learning_curve = train_position_learner(
        total_games,
        player_attack_range=PLAYER_ATTACK_RANGE,
        bot_attack_range=BOT_ATTACK_RANGE,
    )

    print(
        f"Bot fixed starting position: {BOT_START}; "
        f"player range: {PLAYER_ATTACK_RANGE}; bot range: {BOT_ATTACK_RANGE}"
    )
    print("Position results (ranked by average reward: win=1, draw=0.5, loss=0):")
    for position in sorted(values, key=values.get, reverse=True):
        games_played = counts[position]
        wins = records[position]["A"]
        draws = records[position]["draw"]
        print(
            f"  {position}: reward={values[position]:.3f}, "
            f"wins={wins / games_played:.1%}, draws={draws / games_played:.1%}, "
            f"games={games_played}"
        )   

    best_position = max(values, key=values.get)
    print(f"Learner's best starting position: {best_position}")

    plt.plot(range(1, total_games + 1), learning_curve)
    plt.xlabel("Games played")
    plt.ylabel("Average reward (last 100 games)")
    plt.title("Position learner's learning curve")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()