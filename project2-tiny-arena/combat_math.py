import math

def calculate_damage(atk: float, armor: float) -> float:
    """
    TFT Armor formula: atk * (100 / (100 + max(armor, 0)))
    """
    return atk * (100 / (100 + max(armor, 0)))

def apply_damage(current_hp:float, incoming_damage:float) -> float:
    """
    Subtract incoming_damage from current_hp, clamped at 0.0 minimum.
    """
    return max(current_hp - incoming_damage, 0.0)

def test_combat_math():
    # 0 damage -> no change (100 hp, 0 damage == 100 hp)
    assert apply_damage(100, 0) == 100.0

    # 0 armor -> full damage (50 atk, 0 armor == 50 damage)
    assert math.isclose(calculate_damage(50, 0), 50.0)

    # 100 armor -> exactly 50% damage reduction (100 atk, 100 armor == 50 damage)
    assert math.isclose(calculate_damage(100, 100), 50.0)

    # 200 armor -> diminishing returns (100 atk, 200 armor == 33.33 damage)
    assert math.isclose(calculate_damage(100, 200), 100 / 3, rel_tol=1e-3)

    # HP clamp -> current hp = 20, damage = 50 -> hp clamped to 0.0
    assert apply_damage(20, 50) == 0.0

    print("All tests passed!")

if __name__ == "__main__":
    test_combat_math()

