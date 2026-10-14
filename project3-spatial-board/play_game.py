from board import Unit, make_board, can_place, place, print_board
from combat_spatial import spatial_damage
import random


def place_a(board, unit, r, c):
    b = place(board, unit, r, c)
    return b


def place_b(board, unit, r, c):
    b = place(board, unit, r, c)
    return b


# combat (read r, c) from the board to compute spatial damage
def combat(board, atk, armor, attacker_r, attacker_c, defender_r, defender_c):
    if board[attacker_r][attacker_c] is None or board[defender_r][defender_c] is None:
        return None
    damage = spatial_damage(atk, armor, attacker_r, defender_r)
    return damage


def play_placement_game():
    board = make_board()
    unit_a = Unit("A")
    unit_b = Unit("B")

    r1 = random.randint(0, 2)
    c1 = random.randint(0, 3)
    board = place_a(board, unit_a, r1, c1)

    while True:
        r2 = random.randint(0, 2)
        c2 = random.randint(0, 3)
        if can_place(board, r2, c2):
            board = place_b(board, unit_b, r2, c2)
            break

    hp_a, hp_b = 100, 100
    atk_a = random.uniform(15, 25)
    armor_a = random.uniform(0, 50)
    atk_b = random.uniform(15, 25)
    armor_b = random.uniform(0, 50)

    while hp_a > 0 and hp_b > 0:
        damage_a_to_b = combat(board, atk_a, armor_b, r1, c1, r2, c2)
        damage_b_to_a = combat(board, atk_b, armor_a, r2, c2, r1, c1)
        hp_b -= damage_a_to_b
        hp_a -= damage_b_to_a
        if hp_a <= 0 and hp_b <= 0:
            return "draw"
        if hp_a <= 0:
            return "B"
        if hp_b <= 0:
            return "A"
    return "draw" if hp_a <= 0 and hp_b <= 0 else ("A" if hp_a > 0 else "B")


if __name__ == "__main__":
    wins = {"A": 0, "B": 0, "draw": 0}
    for _ in range(1000):
        winner = play_placement_game()
        wins[winner] += 1
    print(wins, f"A {wins['A']/10:.1f}% B {wins['B']/10:.1f}% draw {wins['draw']/10:.1f}%")
