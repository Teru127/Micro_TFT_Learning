import random
import sys
from pathlib import Path
# make `python project2-tiny-arena/play_game.py` from repo root work in VS Code terminal
sys.path.insert(0, str(Path(__file__).parent))
from models import Unit, BattleState
from battle_turn import take_turn

def random_policy(state):
    return random.choice([0, 1])  # 0: attack, 1: defend

def play_game(policy_a, policy_b, initial_state=None, max_turns=50):
    state = initial_state or BattleState(
        agent=Unit(100, random.uniform(15, 25), random.uniform(0, 50)),
        opp=Unit(100, random.uniform(15, 25), random.uniform(0, 50))
    )
    for turn in range(1, max_turns + 1):
        state = take_turn(state, policy_a(state), policy_b(state))
        a_alive, b_alive = state.agent.is_alive, state.opp.is_alive
        if not a_alive and not b_alive: return "draw", turn, state
        if not a_alive: return "opp", turn, state
        if not b_alive: return "agent", turn, state
    return "draw", max_turns, state

if __name__ == "__main__":
    random.seed(0)
    w = {"agent": 0, "opp": 0, "draw": 0}
    for _ in range(1000):
        w[play_game(random_policy, random_policy)[0]] += 1
    print(w, f"agent {w['agent']/10:.1f}% opp {w['opp']/10:.1f}% draw {w['draw']/10:.1f}%")