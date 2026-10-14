from dataclasses import dataclass
from copy import deepcopy

ROWS, COLS = 3, 4

@dataclass(frozen=True)
class Unit:
    id: str

def make_board():
    return [[None for _ in range(COLS)] for _ in range(ROWS)]

def can_place(board, r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and board[r][c] is None

def place(board, unit, r, c):
    if not can_place(board, r, c):
        return None
    new = deepcopy(board)
    new[r][c] = unit
    return new

def print_board(board):
    for r in range(ROWS):
        print(" ".join(cell.id if cell else "." for cell in board[r]))
        