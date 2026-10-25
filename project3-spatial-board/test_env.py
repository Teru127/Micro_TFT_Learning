import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from env import BOT_START, PLAYER_POSITIONS, TinyArenaEnv


class TinyArenaEnvTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        env = TinyArenaEnv()
        check_env(env)

    def test_reset_and_action_positions(self):
        env = TinyArenaEnv()
        observation, info = env.reset(seed=7)

        self.assertIsInstance(env, gym.Env)
        self.assertEqual(env.action_space.n, 11)
        self.assertEqual(len(PLAYER_POSITIONS), 11)
        self.assertEqual(info["bot_position"], BOT_START)
        self.assertEqual(tuple(np.argwhere(observation == 2)[0]), BOT_START)
        self.assertTrue(env.observation_space.contains(observation))

    def test_step_returns_terminal_combat_result(self):
        env = TinyArenaEnv()
        env.reset(seed=11)
        observation, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn(info["outcome"], {"A", "B", "draw"})
        self.assertEqual(reward, {"A": 1.0, "B": -1.0, "draw": 0.0}[info["outcome"]])
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_seed_repeats_same_combat(self):
        results = []
        for _ in range(2):
            env = TinyArenaEnv()
            env.reset(seed=23)
            results.append(env.step(0)[1:])

        self.assertEqual(results[0], results[1])

    def test_rejects_invalid_action(self):
        env = TinyArenaEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(11)


if __name__ == "__main__":
    unittest.main()
