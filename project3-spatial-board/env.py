"""Gymnasium environment where one action selects a starting cell for combat."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from board import COLS, ROWS, Unit, can_place, make_board, place
from combat_spatial import spatial_damage

BOT_START = (1, 2)
PLAYER_ATTACK_RANGE = 3
BOT_ATTACK_RANGE = 1
# Each action number maps to one open board cell, in row-major order.
PLAYER_POSITIONS = tuple(
    (row, col)
    for row in range(ROWS)
    for col in range(COLS)
    if (row, col) != BOT_START
)


def manhattan_distance(r1, c1, r2, c2):
    """Count horizontal plus vertical steps between two cells."""
    return abs(r1 - r2) + abs(c1 - c2)


def move_one_step_toward(r, c, target_r, target_c, rng):
    """Move one cell toward a target, choosing an axis at random if possible."""
    next_cells = []
    if r != target_r:
        next_cells.append((r + (1 if target_r > r else -1), c))
    if c != target_c:
        next_cells.append((r, c + (1 if target_c > c else -1)))
    return next_cells[int(rng.integers(len(next_cells)))]


def combat(atk, armor, attacker_position, defender_position, attack_range, rng):
    """Return damage if the attacker is close enough; otherwise return zero."""
    attacker_r, attacker_c = attacker_position
    defender_r, defender_c = defender_position
    if manhattan_distance(attacker_r, attacker_c, defender_r, defender_c) > attack_range:
        return 0.0
    return spatial_damage(atk, armor, attacker_r, defender_r, rng)


class TinyArenaEnv(gym.Env):
    """Run one complete battle after the agent chooses a starting cell.

    The agent acts once per episode: choose one of 11 positions, then the
    environment simulates the whole fight and returns its outcome as a reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bot_start=BOT_START,
        player_attack_range=PLAYER_ATTACK_RANGE,
        bot_attack_range=BOT_ATTACK_RANGE,
    ):
        super().__init__()
        # Check that the bot can be placed and both units have a usable range.
        if not can_place(make_board(), *bot_start):
            raise ValueError(f"Invalid bot starting position: {bot_start}")
        if player_attack_range < 1 or bot_attack_range < 1:
            raise ValueError("Attack ranges must be at least 1.")

        self.bot_start = bot_start
        self.player_attack_range = player_attack_range
        self.bot_attack_range = bot_attack_range
        # The bot's cell is unavailable, so it is omitted from A's action list.
        self.player_positions = tuple(
            (row, col)
            for row in range(ROWS)
            for col in range(COLS)
            if (row, col) != bot_start
        )
        # The agent sends an integer from 0 to 10, not a (row, column) pair.
        self.action_space = spaces.Discrete(len(self.player_positions))
        # Observations are 3x4 grids: 0 = empty, 1 = A, 2 = bot B.
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(ROWS, COLS),
            dtype=np.int8,
        )
        self._board = None
        self._has_reset = False
        self._episode_done = False

    def _observation(self):
        """Convert the board into the numeric grid Gymnasium agents expect."""
        observation = np.zeros((ROWS, COLS), dtype=np.int8)
        for row, board_row in enumerate(self._board):
            for col, unit in enumerate(board_row):
                if unit is not None:
                    observation[row, col] = 1 if unit.id == "A" else 2
        return observation

    def reset(self, *, seed=None, options=None):
        """Start a fresh episode and return the board before A is placed.

        Pass a seed to make the random combat stats and combat events repeatable.
        """
        if options:
            raise ValueError("Reset options are not supported.")
        # Gymnasium initializes its random-number generator here.
        super().reset(seed=seed)
        self._board = make_board()
        # B is already on the board; A's cell will be chosen by the next action.
        self._board = place(self._board, Unit("B"), *self.bot_start)
        self._has_reset = True
        self._episode_done = False

        info = {
            "bot_position": self.bot_start,
            # Use this mapping to translate action numbers back into board cells.
            "action_positions": self.player_positions,
        }
        return self._observation(), info

    def _simulate_combat(self, player_position):
        """Run a fight from the chosen cell and return winner and final stats."""
        # Each fighter gets randomized stats; interval is seconds between attacks.
        player_stats = {
            "hp": 100.0,
            "attack": self.np_random.uniform(15.0, 25.0),
            "armor": self.np_random.uniform(0.0, 50.0),
            "attack_interval": 1.0 / self.np_random.uniform(0.8, 1.2),
        }
        bot_stats = {
            "hp": 100.0,
            "attack": self.np_random.uniform(15.0, 25.0),
            "armor": self.np_random.uniform(0.0, 50.0),
            "attack_interval": 1.0 / self.np_random.uniform(0.8, 1.2),
        }
        player_position_now = player_position
        bot_position_now = self.bot_start
        next_player_attack = player_stats["attack_interval"]
        next_bot_attack = bot_stats["attack_interval"]
        next_move = 1.0

        # Advance the battle clock to whichever event happens next.
        while player_stats["hp"] > 0 and bot_stats["hp"] > 0:
            event_time = min(next_player_attack, next_bot_attack, next_move)

            if math.isclose(next_move, event_time, abs_tol=1e-9):
                # Both fighters move toward the opponent if outside their own range.
                # Randomizing the order prevents always giving the same unit first move.
                if bool(self.np_random.integers(2)):
                    if manhattan_distance(*player_position_now, *bot_position_now) > self.player_attack_range:
                        player_position_now = move_one_step_toward(
                            *player_position_now, *bot_position_now, self.np_random
                        )
                    if manhattan_distance(*player_position_now, *bot_position_now) > self.bot_attack_range:
                        bot_position_now = move_one_step_toward(
                            *bot_position_now, *player_position_now, self.np_random
                        )
                else:
                    if manhattan_distance(*player_position_now, *bot_position_now) > self.bot_attack_range:
                        bot_position_now = move_one_step_toward(
                            *bot_position_now, *player_position_now, self.np_random
                        )
                    if manhattan_distance(*player_position_now, *bot_position_now) > self.player_attack_range:
                        player_position_now = move_one_step_toward(
                            *player_position_now, *bot_position_now, self.np_random
                        )
                next_move += 1.0

            # Process attacks due at this time; both can attack in the same event.
            player_attacks = math.isclose(
                next_player_attack, event_time, abs_tol=1e-9
            )
            bot_attacks = math.isclose(next_bot_attack, event_time, abs_tol=1e-9)

            if player_attacks:
                bot_stats["hp"] -= combat(
                    player_stats["attack"],
                    bot_stats["armor"],
                    player_position_now,
                    bot_position_now,
                    self.player_attack_range,
                    self.np_random,
                )
                next_player_attack += player_stats["attack_interval"]
            if bot_attacks:
                player_stats["hp"] -= combat(
                    bot_stats["attack"],
                    player_stats["armor"],
                    bot_position_now,
                    player_position_now,
                    self.bot_attack_range,
                    self.np_random,
                )
                next_bot_attack += bot_stats["attack_interval"]

        # A simultaneous knockout is a draw; otherwise the surviving fighter wins.
        if player_stats["hp"] <= 0 and bot_stats["hp"] <= 0:
            outcome = "draw"
        elif player_stats["hp"] > 0:
            outcome = "A"
        else:
            outcome = "B"

        # Keep the final board available as the observation returned by step().
        self._board = make_board()
        self._board = place(self._board, Unit("A"), *player_position_now)
        self._board = place(self._board, Unit("B"), *bot_position_now)
        return outcome, player_stats, bot_stats, player_position_now, bot_position_now

    def step(self, action):
        """Simulate combat for an action and return Gymnasium's five results.

        Reward: +1 for an A win, 0 for a draw, and -1 for a bot win.
        This environment ends after this single action and battle.
        """
        # Require a fresh episode before accepting an action.
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode is finished; call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer from 0 to {self.action_space.n - 1}.")

        # Convert the chosen action index into A's starting (row, column).
        player_position = self.player_positions[int(action)]
        outcome, player, bot, final_player_position, final_bot_position = (
            self._simulate_combat(player_position)
        )
        reward = {"A": 1.0, "draw": 0.0, "B": -1.0}[outcome]
        self._episode_done = True

        # Keep extra battle details here; the agent can inspect them for analysis.
        info = {
            "outcome": outcome,
            "player_start": player_position,
            "bot_start": self.bot_start,
            "final_positions": {
                "A": final_player_position,
                "B": final_bot_position,
            },
            "remaining_hp": {
                "A": max(player["hp"], 0.0),
                "B": max(bot["hp"], 0.0),
            },
        }
        # terminated=True because this one-action episode is complete;
        # truncated=False because it did not stop due to an external time limit.
        return self._observation(), reward, True, False, info
