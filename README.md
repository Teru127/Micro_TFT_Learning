# Micro TFT Learning

Build a Teamfight Tactics agent that learns by **playing itself**, not by imitating human play. Same recipe Riot described publicly at GDC (Ran Cao, "Simulating Teamfight Tactics Using Deep Learning for Fast Reinforcement Learning AI Training").

This repo is a **project-based learning path**. Each project is a small, runnable artifact that adds exactly one concept. The goal is a self-play TFT agent, but the real value is in the ladder.

## Stack

```
env you write  ->  PPO / tabular RL  ->  self-play  ->  scaled TFT
                    (stable-baselines3)  (PettingZoo)   (Project 6)
```

## Progress

| # | Project | Status |
|---|---|---|
| 0 | Tabular Q-learning, Tic-Tac-Toe | done |
| 1 | PPO on CartPole | done |
| 2 | TinyArena, first custom Gym env | done |
| 3 | Spatial board, positioning | in progress |
| 4 | Economy layer | planned |
| 5 | Multi-agent self-play | planned |
| 6 | Scale to full TFT | planned |

---

## Project 0: Tabular Q-learning on Tic-Tac-Toe

A self-play agent with a plain Q-table, no neural net.

`project0-tictactoe/tic-tac-toe.py`

**Teaches:** the `state -> action -> reward -> update` loop by hand, plus self-play value semantics. A Q-value describes a move for whoever is about to make it, so the next state's value gets negated (the opponent moves next).

**Ships:** an agent that draws against perfect minimax play. Tests assert a hand-computed Q-update (`tic-tac-toe.py:261`).

---

## Project 1: PPO on a standard Gym env

Swap the hand-written loop for Stable-Baselines3 PPO on CartPole-v1.

| File | Purpose |
|---|---|
| `project1-cartpole-ppo/cartpole_ppo.py` | training |
| `project1-cartpole-ppo/evaluate_agent.py` | deterministic eval, mean and std |
| `project1-cartpole-ppo/plot_curve.py` | reward curve from the Monitor CSV |

**Teaches:** reading a training curve to tell "not learning" apart from "learning slowly".

**Result:** `monitor.csv` ends at **500.0**, the environment maximum. Solved.

---

## Project 2: TinyArena, your first Gym environment

A 2-unit, fixed-position, no-movement 1v1 auto-battle. No existing environment to lean on, so you write the `reset`/`step` contract yourself.

| File | Purpose |
|---|---|
| `combat_math.py` | TFT armor formula, `damage = atk * 100 / (100 + armor)` |
| `models.py` | immutable `Unit` and `BattleState` dataclasses |
| `battle_turn.py` | one simultaneous tick (attack or defend) |
| `play_game.py` | full loop outside Gym, 1000 random-vs-random games |
| `env.py` | `TinyArenaEnv(gym.Env)`, `Box` obs, `Discrete(2)` action |
| `eval_random.py` | baseline number to beat |
| `train.py`, `plot.py` | PPO training and curve |

**Teaches:** the Gymnasium API contract (`reset`/`step`, `observation_space`, `action_space`, and the difference between `terminated` and `truncated`), reward shaping, and `Monitor` logging.

**Ships:** PPO clears the random baseline by a wide margin.

| Policy | mean reward | std | win rate |
|---|---|---|---|
| random baseline (200 eps) | `-0.24` | `9.58` | `50.5%` |
| **PPO deterministic (200 eps)** | **`8.08`** | `4.50` | **`91.5%`** |

Training curve goes from `-0.2` to `8.3` over 100k steps.

![TinyArena training curve](project2-tiny-arena/curve.png)

---

## Project 3: Spatial board, positioning matters

Add a 3x4 board so **where you place** becomes the decision.

| File | Purpose |
|---|---|
| `board.py` | grid, `can_place`, immutable `place`, `print_board` |
| `combat_spatial.py` | `spatial_damage` with a row factor (back row takes `0.85x` with jitter) |
| `play_game.py` | random placement, then combat to death |

**Teaches:** state representation for spatial problems, and why the signal must exist in the environment *before* an agent can learn to exploit it.

**Verified so far:**

* Position matters. Back row wins **86.4%** vs front row with randomized stats (`864/1000`).
* Random placement is unbiased. **41.6% / 38.8% / 19.6% draw**. Draws run high because damage resolves simultaneously.

**Ships, not yet:** an agent whose placement heatmap is visibly backline-biased against a uniform random one.

---

## Roadmap

| # | Project | Teaches |
|---|---|---|
| 4 | Economy layer (gold, shop, level) | Hierarchical and decomposed agents, credit assignment over turns |
| 5 | Multi-agent self-play (PettingZoo, 8 players) | Non-stationary opponents, opponent pools |
| 5.5 | Neural combat surrogate `P(win \| A, B)` | Why tick-by-tick sims are too slow for self-play (the GDC trick) |
| 6 | Scale to full TFT (items, champion pool) | Pure engineering, no new concepts |

## Run

```powershell
.venv\Scripts\Activate.ps1

python project0-tictactoe\tic-tac-toe.py
python project1-cartpole-ppo\cartpole_ppo.py
python project2-tiny-arena\train.py
python project2-tiny-arena\plot.py
python project3-spatial-board\test_board.py
```

## Notes

* Offline only. Nothing here touches the game client, and automating the real client violates Riot's ToS.
* `*.zip`, `*.csv`, and `logs/` are gitignored. Trained models and monitor data are reproducible, not source.
* TFT is a trademark of Riot Games. This is an independent, fan-made project.