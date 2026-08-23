# Micro-TFT-Learning

A project-based learning ladder toward building a self-play reinforcement
learning agent for Teamfight Tactics -- inspired by Riot Games' own GDC talk
on training an internal TFT-playing AI for balance testing.

Each project is a complete, working thing, not a tutorial exercise -- code
from each one carries forward into the next. The goal is to learn RL by
hitting real bugs and diagnosing them, not by reading about them.

## Progress

| # | Project | Status | What it proves |
|---|---------|--------|-----------------|
| 0 | [Tabular Q-learning, Tic-Tac-Toe (self-play)](./project0-tictactoe) | Done | Hand-implemented the Bellman update, self-play, and a from-scratch minimax opponent for evaluation |
| 1 | [PPO on CartPole (Stable-Baselines3)](./project1-cartpole-ppo) | Done | Same core RL ideas, now with a library and a neural net handling the mechanics |
| 2 | Custom Gym environment, 2-unit fixed-position battler | Planned | Building the Gym API contract from scratch |
| 3 | Add a spatial action space (positioning) | Planned | CNN-based policy over a hex/grid board |
| 4 | Add an economy/shop layer as a second agent | Planned | Hierarchical decomposition, credit assignment |
| 5 | Real multi-agent self-play (PettingZoo, opponent pooling) | Planned | The actual target milestone for this ladder |
| 6 | Scale toward full TFT rules | Stretch goal | Not committed to -- time/compute permitting |

## Why stop planning at Project 5

Project 5 (multi-agent self-play with a decomposed agent) is already a
complete, demonstrable RL portfolio piece on its own. Project 6 -- full TFT
scale -- is a much larger engineering effort closer to what a dedicated team
does, and is treated here as a stretch goal, not a commitment.

## A running list of real lessons, not just definitions

- **A single training run is a random sample, not a guarantee** -- identical
  code produced very different results run to run until training volume and
  exploration schedules were fixed properly (Project 0).
- **Self-play agents can converge to a policy that looks correct against
  itself but has exploitable blind spots** -- discovered by testing against a
  true adversary (minimax), not just against a copy of the agent itself
  (Project 0).
- **A library hides real mechanics you should know about anyway** --
  `total_timesteps` in Stable-Baselines3 is a minimum, not an exact count,
  because of how rollout collection works under the hood (Project 1).

See each project's own README for the full debugging story.