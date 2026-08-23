# Project 0 — Tabular Q-Learning Tic-Tac-Toe (Self-Play)

First project in a learning ladder toward a self-play RL agent for Teamfight Tactics.
Goal: implement the state -> action -> reward -> update loop by hand, with a state
space small enough to fully understand, before touching any library.

## Ship criterion
Trained agent must **never lose** to a perfect (minimax) opponent, evaluated with
zero exploration, playing both marks (X and O).

## Result
- **vs. minimax (200 games, both marks):** 0 wins / 200 draws / 0 losses
- **vs. random opponent (100,000 games):** 86,027 wins / 13,584 draws / 389 losses (0.39% loss rate)

## What's actually in here
- Tabular Q-learning with epsilon-greedy exploration
- Self-play training with a negamax-style perspective flip (a Q-value means
  "good for whoever is about to move," so the bootstrap target for the
  opponent's resulting position gets negated)
- A from-scratch minimax opponent, used only for evaluation, never training
- A swappable evaluation harness (`evaluate(q_table, opponent_fn, n_games)`)

## Bugs found and fixed along the way
- **Turn-order bugs** from conflating "who started the game" (random, set once)
  with "whose turn is it now" (deterministic, derived every move) -- caught by
  tracing individual games move-by-move, not by looking at aggregate stats.
- **Global-state fragility** in an early `whose_turn` implementation -- fixed by
  passing `starting_player` explicitly instead of relying on module-level state.
- **A self-play blind spot**: the agent converged to ~99% draws against itself
  but still lost consistently to a perfect opponent. Root cause: since the same
  policy plays both sides, it rarely explores openings it doesn't already favor
  (e.g. edge openings), so it never learns to defend them -- a real, general
  self-play exploitability issue, not a bug in the update rule. Fixed by mixing
  the minimax solver in as an occasional training opponent (~50% of games),
  which finally exposed the agent to lines it would never choose for itself.
- **Training run flakiness**: identical code, different random seeds, produced
  wildly different results (0 losses one run, 100/200 losses the next) at low
  training volume -- a reminder that a single RL training run is a sample, not
  a guarantee.

## How to run
```
python3 tictactoe_qlearning.py
```