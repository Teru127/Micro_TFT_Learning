import math
from models import Unit, BattleState
from battle_turn import take_turn

def test_exact_math_both_attack():
    initial_state = BattleState(
        agent=Unit(hp=100, atk=20, armor=0),
        opp=Unit(hp=100, atk=20, armor=0)
    )

    next_state = take_turn(initial_state, action_a = 0, action_b = 0)

    assert math.isclose(next_state.opp.hp, 80.0, abs_tol=1e-5)
    assert math.isclose(next_state.agent.hp, 80.0, abs_tol=1e-5)
    print("test 1 passed.")

def test_defend_double_armor():
    initial_state = BattleState(
        agent=Unit(hp=100, atk=20, armor=50),
        opp=Unit(hp=100, atk=100, armor=0)
    )

    next_state = take_turn(initial_state, action_a = 1, action_b = 0)

    assert math.isclose(next_state.opp.hp, 100.0, abs_tol=1e-5)  # Agent defended, Opponent should not take damage
    assert math.isclose(next_state.agent.hp, 50.0, abs_tol=1e-5)  # Agent defended, Opponent should deal less damage
    assert math.isclose(next_state.agent.armor, 50.0, abs_tol=1e-5)  # Agent's armor should remain the same after defending
    print("test 2 passed.")

def test_one_hit():
    initial_state = BattleState(
        agent=Unit(hp=100, atk=200, armor=0),
        opp=Unit(hp=100, atk=20, armor=0)
    )

    next_state = take_turn(initial_state, action_a = 0, action_b = 0)

    assert math.isclose(next_state.opp.hp, 0.0, abs_tol=1e-5)  # Opponent should be defeated
    assert math.isclose(next_state.agent.hp, 80.0, abs_tol=1e-5)  # Agent should take damage from Opponent
    print("test 3 passed.")

if __name__ == "__main__":
    test_exact_math_both_attack()
    test_defend_double_armor()
    test_one_hit()