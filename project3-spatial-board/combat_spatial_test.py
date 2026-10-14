from combat_spatial import spatial_damage
import random
wins_back = 0
for _ in range(1000):
    atk = random.uniform(15, 25)
    armor = random.uniform(0, 50)
    pos_x = random.randint(0, 2)
    hp_back, hp_front = 100, 100
    while hp_back > 0 and hp_front > 0:
        hp_back -= spatial_damage(atk, armor, pos_x, 2)
        hp_front -= spatial_damage(atk, armor, pos_x, 0)
    if hp_back > 0: wins_back += 1  
print(f"Back defender wins: {wins_back}/1000")