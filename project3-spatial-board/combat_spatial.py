import sys
import random
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "project2-tiny-arena"))
from combat_math import calculate_damage

def position_factor(defender_r: int, rng=None) -> float:
    # back row = 2 take less damage
    if defender_r == 2:
        rng = random if rng is None else rng
        return 0.85 * rng.uniform(0.9, 1.15)
    return 1.0


def spatial_damage(atk, armor, attacker_r, defender_r, rng=None):
    base_damage = calculate_damage(atk, armor)
    factor = position_factor(defender_r, rng)
    return base_damage * factor 
