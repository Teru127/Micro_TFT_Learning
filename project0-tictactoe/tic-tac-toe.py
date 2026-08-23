import random


# Q-learning
q_table = {}


def get_q_value(state, action, q_table):
    return q_table.get((state, action), 0)


def choose_action(board, q_table, epsilon):
    legal_moves = available_moves(board)
    if not legal_moves:
        return None

    state = tuple(board)
    if random.random() < epsilon:
        return random.choice(legal_moves)

    q_values = [get_q_value(state, action, q_table) for action in legal_moves]
    highest_value = max(q_values)
    best_moves = [
        action for action, value in zip(legal_moves, q_values)
        if value == highest_value
    ]
    return random.choice(best_moves)


def get_reward_for_player(player, outcome):
    if outcome == "Draw":
        return 0
    return 1 if player == outcome[0] else -1


def train_on_game(q_table, history, outcome, alpha=0.3, gamma=0.95):
    for index in range(len(history) - 1, -1, -1):
        state, action, player = history[index]

        if index == len(history) - 1:
            target = get_reward_for_player(player, outcome)
        else:
            next_state = history[index + 1][0]
            next_values = [
                get_q_value(next_state, next_action, q_table)
                for next_action in available_moves(next_state)
            ]
            target = -gamma * max(next_values, default=0)

        old_value = get_q_value(state, action, q_table)
        q_table[(state, action)] = old_value + alpha * (target - old_value)


# Board and game rules
def newboard():
    return [None, None, None, None, None, None, None, None, None]


def start_first():
    return random.choice(['X', 'O'])


def whose_turn(board, starting_player):
    total_moves = board.count('X') + board.count('O')
    other_player = 'O' if starting_player == 'X' else 'X'
    return starting_player if total_moves % 2 == 0 else other_player


def available_moves(board):
    return [index for index, square in enumerate(board) if square is None]


def make_move(board, position, player):
    if board[position] is None:
        board[position] = player
        return True
    return False


def check_winner(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for first, second, third in winning_combinations:
        if board[first] == board[second] == board[third]:
            if board[first] is not None:
                return board[first]
    return None


def check_draw(board):
    return all(square is not None for square in board) and check_winner(board) is None


def game_status(board):
    winner = check_winner(board)
    if winner:
        return f"{winner} wins!"
    if check_draw(board):
        return "Draw"
    return "In progress"


# Evaluation
minimax_cache = {}


def minimax(board, player):
    state = tuple(board)
    cache_key = (state, player)
    if cache_key in minimax_cache:
        return minimax_cache[cache_key]

    outcome = game_status(board)
    if outcome != "In progress":
        if outcome == "Draw":
            value = 0
        elif outcome.startswith(f"{player} wins"):
            value = 1
        else:
            value = -1
        result = (None, value)
        minimax_cache[cache_key] = result
        return result

    opponent = 'O' if player == 'X' else 'X'
    move_values = []
    for move in available_moves(board):
        next_board = board[:]
        make_move(next_board, move, player)
        _, opponent_value = minimax(next_board, opponent)
        move_values.append((move, -opponent_value))

    best_value = max(value for _, value in move_values)
    best_moves = [
        move for move, value in move_values
        if value == best_value
    ]
    result = (random.choice(best_moves), best_value)
    minimax_cache[cache_key] = result
    return result


def random_opponent(board):
    return random.choice(available_moves(board))


def minimax_opponent(board):
    x_moves = board.count('X')
    o_moves = board.count('O')
    player = 'X' if x_moves == o_moves else 'O'
    move, _ = minimax(board, player)
    return move


def evaluate(q_table, opponent_fn, n_games=100000):
    results = {"wins": 0, "draws": 0, "losses": 0}

    for game_number in range(n_games):
        board = newboard()
        agent_player = 'X' if game_number % 2 == 0 else 'O'
        current_player = 'X'

        while True:
            if current_player == agent_player:
                move = choose_action(board, q_table, epsilon=0)
            else:
                move = opponent_fn(board)

            make_move(board, move, current_player)
            outcome = game_status(board)
            if outcome != "In progress":
                if outcome == "Draw":
                    results["draws"] += 1
                elif outcome.startswith(f"{agent_player} wins"):
                    results["wins"] += 1
                else:
                    results["losses"] += 1
                break

            current_player = 'O' if current_player == 'X' else 'X'

    return results


# Self-play training
def self_play_simulation(
    q_table,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9999,
    minimax_probability=0.5,
):
    board = newboard()
    starting_player = start_first()
    current_player = starting_player
    history = []
    minimax_player = None
    if random.random() < minimax_probability:
        minimax_player = random.choice(['X', 'O'])

    while True:
        if current_player == minimax_player:
            move, _ = minimax(board, current_player)
        else:
            move = choose_action(board, q_table, epsilon)
        history.append((tuple(board), move, current_player))
        make_move(board, move, current_player)

        outcome = game_status(board)
        if outcome != "In progress":
            train_on_game(q_table, history, outcome)
            return outcome

        current_player = whose_turn(board, starting_player)


def train_self_play(
    q_table,
    n_games=10000,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9999,
    minimax_probability=0.5,
):
    for _ in range(n_games):
        self_play_simulation(
            q_table,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            minimax_probability=minimax_probability,
        )
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return epsilon


# Tests
def test_game_status():
    assert game_status(['X', 'X', 'X', 'O', 'O', None, None, None, None]) == "X wins!"
    assert game_status(['O', 'O', 'O', 'X', 'X', None, None, None, None]) == "O wins!"


def test_choose_action():
    board = newboard()
    assert get_q_value(tuple(board), 0, {}) == 0
    assert choose_action(board, {}, 1) in available_moves(board)
    assert choose_action(board, {(tuple(board), 4): 10}, 0) == 4
    assert choose_action(draw_board(), {}, 0) is None


def test_train_on_game():
    q_values = {}
    first_state = tuple(newboard())
    second_state = ('X', None, None, None, None, None, None, None, None)
    history = [(first_state, 0, 'X'), (second_state, 1, 'O')]

    train_on_game(q_values, history, "X wins!")
    assert q_values[(second_state, 1)] == -0.3


def draw_board():
    return ['X', 'O', 'X', 'X', 'O', 'O', 'O', 'X', 'X']


def test_evaluate():
    q_values = {}
    results = evaluate(q_values, random_opponent, n_games=4)
    assert sum(results.values()) == 4
    assert q_values == {}


def test_minimax():
    board = ['X', 'X', None,
             'O', 'O', None,
             'X', None, None]
    move, value = minimax(board, 'X')
    assert move == 2
    assert value == 1

    move, value = minimax(newboard(), 'X')
    assert move in available_moves(newboard())
    assert value == 0


def run_tests():
    test_game_status()
    test_choose_action()
    test_train_on_game()
    test_evaluate()
    test_minimax()
    print("All tests passed!")


def main():
    run_tests()
    train_self_play(q_table, n_games=500000, minimax_probability=0.5)
    results = evaluate(q_table, random_opponent)
    print(f"Evaluation against random opponent: {results}")
    minimax_results = evaluate(q_table, minimax_opponent)
    print(f"Evaluation against minimax: {minimax_results}")


if __name__ == "__main__":
    main()
