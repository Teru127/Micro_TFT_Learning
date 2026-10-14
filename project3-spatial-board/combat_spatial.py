import sys
import random
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "project2-tiny-arena"))
from combat_math import calculate_damage

def position_factor(defender_r: int) -> float:
    # back row = 2 take less damage
    return 0.85 * random.uniform(0.9, 1.15) if defender_r == 2 else 1.0

def spatial_damage(atk, armor, attacker_r, defender_r):
    base_damage = calculate_damage(atk, armor)
    factor = position_factor(defender_r)
    return base_damage * factor 
