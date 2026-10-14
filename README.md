# Micro TFT Learning

Build a Teamfight Tactics agent that learns by **playing itself** — not by imitating human play. Same recipe Riot described publicly at GDC (Ran Cao, "Simulating Teamfight Tactics Using Deep Learning for Fast Reinforcement Learning AI Training").

This repo is a **project-based learning path**. Each project is a small, runnable artifact that adds one concept. The final goal is a self-play TFT agent — but the value is in the ladder, not the destination.

## Stack

```
Gymnasium env  ->  PPO / tabular RL  ->  self-play  ->  scaled TFT
 (you write)       (stable-baselines3)   (PettingZoo)    (Project 6)
```

---

## Project 0 — Tabular Q-learning on Tic-Tac-Toe

Build a self-play agent with a plain Q-table (no neural net).

* `project0-tictactoe/tic-tac-toe.py`

**Teaches:** the `state -> action -> reward -> update` loop by hand; self-play value semantics (a Q-value describes *the move*, so the next state's value is negated since the opponent moves next).

**Ships:** an agent that draws against a perfect minimax opponent. Tests assert a hand-computed Q-update (`tic-tac-toe.py:261`).

---

## Project 1 — PPO on a standard Gym env

Replace the hand-written loop with Stable-Baselines3 PPO on CartPole-v1.

* `project1-cartpole-ppo/cartpole_ppo.py` — training
* `project1-cartpole-ppo/evaluate_agent.py` — deterministic eval, mean ± std
* `project1-cartpole-ppo/plot_curve.py` — reward curve from `Monitor` CSV

**Teaches:** reading a training curve to diagnose "not learning" vs "learning slowly".

**Result:** `monitor.csv` ends at **500.0** reward (env max) — solved.

---

## Project 2 — TinyArena: your first Gym environment ✅

A 2-unit, fixed-position, no-movement 1v1 auto-battle. No existing environment — you write the `reset`/`step` contract yourself.

* `combat_math.py` — TFT armor formula `damage = atk * 100 / (100 + armor)`
* `models.py` — immutable `Unit` / `BattleState` dataclasses
* `battle_turn.py` — one simultaneous tick (`attack` or `defend`)
* `play_game.py` — full loop outside Gym, 1000 random-vs-random games
* `env.py` — `TinyArenaEnv(gym.Env)`, `Box` obs / `Discrete(2)` action
* `eval_random.py` — baseline number
* `train.py`, `plot.py`, `curve.png` — PPO training + curve

**Teaches:** the Gymnasium API contract (`reset`/`step`, `observation_space`, `action_space`, `terminated` vs `truncated`); reward shaping; `Monitor` logging.

**Ships:** PPO beats the random baseline by a clear margin.

| | mean reward | std | win rate |
|---|---|---|---|
| random baseline (200 eps) | `-0.24` | `9.58` | `50.5%` |
| **PPO deterministic (200 eps)** | **`8.08`** | `4.50` | **`91.5%`** |

Training curve (`curve.png`): `-0.2 -> 8.3` over 100k steps.

![TinyArena training curve](project2-tiny-arena/curve.png)

---

## Project 3 — Spatial board: positioning matters (in progress)

Add a `3x4` board so **where you place** becomes the decision.

* `board.py` — grid, `can_place`, immutable `place`, `print_board`
* `combat_spatial.py` — `spatial_damage` with a row factor (back row takes `0.85x * jitter`)
* `play_game.py` — random placement + combat to death

**Teaches:** state representation for spatial problems; why a signal must exist in the environment *before* an agent can learn to exploit it.

**Verified so far:**
* Position matters: back row wins **86.4%** vs front row with randomized stats (`864/1000`)
* Random-vs-random placement is unbiased: **41.6% / 38.8% / 19.6% draw** — draws are high because damage is simultaneous

**Ships (not yet):** an agent whose placement heatmap is visibly backline-biased vs uniform random.

---

## Roadmap

| # | Project | Teaches |
|---|---|---|
| 4 | Economy layer (gold, shop, level) | Hierarchical/decomposed agents, credit assignment over turns |
| 5 | Multi-agent self-play (PettingZoo, 8 players) | Non-stationary opponents, opponent pools |
| 5.5 | Neural combat surrogate `P(win \| A, B)` | Why tick-by-tick sims are too slow for self-play (the GDC trick) |
| 6 | Scale to full TFT (items, champion pool) | Pure engineering — no new concepts |

## Run

```powershell
# activate env first
.venv\Scripts\Activate.ps1

python project0-tictactoe\tic-tac-toe.py
python project1-cartpole-ppo\cartpole_ppo.py
python project2-tiny-arena\train.py
python project2-tiny-arena\plot.py
python project3-spatial-board\test_board.py
```

## Notes

* Offline only — nothing here touches the game client. Automating the real client violates Riot's ToS.
* `*.zip`, `*.csv`, `logs/` are gitignored (trained models and monitor data are reproducible, not source).
* TFT is a trademark of Riot Games. Independent, fan-made project.