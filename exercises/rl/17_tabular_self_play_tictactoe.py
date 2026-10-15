"""
============================================================
EXERCISE 17: TABULAR SELF PLAY TICTACTOE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny self-play agent for Tic-Tac-Toe.

This exercise uses real game data.
The data comes from legal Tic-Tac-Toe games played by the agent against itself.
Do not use made-up random board tensors.

WHY THIS EXERCISE EXISTS
AlphaGo-style systems start with a simple idea:
- play games
- store real game states
- learn which states lead to wins

This file teaches that idea with a small game you can fully inspect.

DATA
Each board is a real Tic-Tac-Toe state.

Board values:
- `0`: empty cell
- `1`: X
- `-1`: O

Board shape:
- list of 9 integers

Example:

```text
[ 1,  0, -1,
  0,  1,  0,
 -1,  0,  0]
```

STEP-BY-STEP TODO
1. Build legal move and winner checks.
2. Play self-play games.
3. Store each visited state.
4. Update a value table after the final result.
5. Evaluate against a random legal player.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


Board = tuple[int, ...]
ValueTable = dict[Board, float]


@dataclass
class TicTacToeConfig:
    """Settings for tabular Tic-Tac-Toe self-play."""

    self_play_games: int = 5000
    eval_games: int = 200
    epsilon: float = 0.2
    learning_rate: float = 0.1
    random_state: int = 42


def empty_board() -> Board:
    """Return an empty Tic-Tac-Toe board."""
    # Goal:
    # Return the starting board before any player has moved.
    #
    # Output:
    # - `Board`
    # - This is a tuple with 9 values.
    # - Every value should be `0`.
    #
    # Board index layout:
    # ```text
    # 0 | 1 | 2
    # 3 | 4 | 5
    # 6 | 7 | 8
    # ```
    #
    # Method:
    # Use:
    # `tuple([0] * 9)`
    #
    # Checks:
    # - `len(board)` should be 9.
    # - `set(board)` should be `{0}`.
    raise NotImplementedError("Return an empty Tic-Tac-Toe board.")


def legal_moves(board: Board) -> list[int]:
    """Return legal cell indexes."""
    # Goal:
    # Return the indexes where a player can place a mark.
    #
    # Input:
    # - `board`: Board
    # - Shape: 9 cells.
    # - Values: `0` means empty, `1` means X, `-1` means O.
    #
    # Output:
    # - `list[int]`
    # - Each value is an index from `0` to `8`.
    #
    # Method:
    # A move is legal when:
    # `board[index] == 0`
    #
    # Steps:
    # 1. Create an empty `moves` list.
    # 2. Loop over indexes `0` through `8`.
    # 3. If `board[index] == 0`, add the index to `moves`.
    # 4. Return `moves`.
    #
    # Checks:
    # - For an empty board, this should return `[0, 1, 2, 3, 4, 5, 6, 7, 8]`.
    # - For board `(1, 0, -1, 0, 0, 0, 0, 0, 0)`, this should not include `0` or `2`.
    raise NotImplementedError("Return legal Tic-Tac-Toe moves.")


def apply_move(board: Board, move: int, player: int) -> Board:
    """Return a new board after one legal move."""
    # Goal:
    # Place `player` at `move` and return a new board.
    # Do not change the old board.
    #
    # Input:
    # - `board`: Board with 9 cells
    # - `move`: integer from 0 to 8
    # - `player`: `1` for X or `-1` for O
    #
    # Output:
    # - `Board`
    # - New board tuple with the move applied.
    #
    # Example:
    # If `board = (0, 0, 0, 0, 0, 0, 0, 0, 0)`, `move = 4`, and `player = 1`,
    # the result should be `(0, 0, 0, 0, 1, 0, 0, 0, 0)`.
    #
    # Method:
    # Use:
    # - `cells = list(board)`
    # - `cells[move] = player`
    # - `return tuple(cells)`
    #
    # Steps:
    # 1. Check that `player` is either `1` or `-1`.
    # 2. Check that `move` is in `legal_moves(board)`.
    # 3. Convert the board to a list.
    # 4. Write the player mark into the chosen cell.
    # 5. Convert the list back to a tuple and return it.
    #
    # Checks:
    # - Do not change the old board in place.
    # - Raise `ValueError` if the move is not legal.
    # - Raise `ValueError` if `player` is not `1` or `-1`.
    raise NotImplementedError("Apply one legal Tic-Tac-Toe move.")


def winner(board: Board) -> int:
    """Return the winner, or 0 when there is no winner yet."""
    # Goal:
    # Detect whether X or O has three marks in a row.
    # This function only checks for a winner. It does not decide whether the board is a draw.
    #
    # Input:
    # - `board`: Board with 9 cells.
    #
    # Output:
    # - `1` if X won
    # - `-1` if O won
    # - `0` if nobody has won yet
    #
    # Method:
    # - Check these index triples:
    #   `(0, 1, 2)`, `(3, 4, 5)`, `(6, 7, 8)`,
    #   `(0, 3, 6)`, `(1, 4, 7)`, `(2, 5, 8)`,
    #   `(0, 4, 8)`, `(2, 4, 6)`.
    #
    # For each triple:
    # 1. Read the three board values.
    # 2. If all three are `1`, return `1`.
    # 3. If all three are `-1`, return `-1`.
    # 4. If no triple wins, return `0`.
    #
    # Checks:
    # - Board `(1, 1, 1, 0, 0, 0, 0, 0, 0)` should return `1`.
    # - Board `(-1, 0, 0, -1, 0, 0, -1, 0, 0)` should return `-1`.
    # - Empty board should return `0`.
    raise NotImplementedError("Return the Tic-Tac-Toe winner.")


def choose_move(board: Board, player: int, values: ValueTable, epsilon: float) -> int:
    """Choose a legal move with epsilon-greedy play."""
    # Goal:
    # Choose one legal move.
    # Sometimes explore with a random move.
    # Otherwise choose the move with the best value estimate.
    #
    # Input:
    # - `board`: Board
    # - `player`: `1` for X or `-1` for O
    # - `values`: maps board states to values from X's point of view
    # - `epsilon`: chance of choosing a random legal move
    #
    # Output:
    # - `int`
    # - A legal move index from `legal_moves(board)`.
    #
    # Value meaning:
    # - High value is good for X.
    # - Low value is good for O.
    # - Unknown boards should use value `0.0`.
    #
    # Method:
    # Use:
    # - `moves = legal_moves(board)`
    # - `random.random() < epsilon` for exploration
    # - `random.choice(moves)` for a random legal move
    # - `values.get(next_board, 0.0)` for unknown board values
    #
    # For each candidate move:
    # - build `next_board = apply_move(board, move, player)`
    # - read `next_value = values.get(next_board, 0.0)`
    #
    # Rule:
    # - If player is X (`1`), choose the move with the highest `next_value`.
    # - If player is O (`-1`), choose the move with the lowest `next_value`.
    #
    # Steps:
    # 1. Get legal moves.
    # 2. If there are no legal moves, raise `ValueError`.
    # 3. With probability `epsilon`, return a random legal move.
    # 4. Score every legal move by the value of the next board.
    # 5. Choose highest value for X or lowest value for O.
    # 6. Return the chosen move.
    #
    # Checks:
    # - The returned move must be in `legal_moves(board)`.
    # - With `epsilon=1.0`, the function should explore.
    # - With `epsilon=0.0`, the function should use the value table.
    raise NotImplementedError("Choose one Tic-Tac-Toe move.")


def play_self_play_game(values: ValueTable, config: TicTacToeConfig) -> tuple[list[Board], int]:
    """Play one self-play game and return visited states and result."""
    # Goal:
    # Generate one real legal Tic-Tac-Toe game by letting the agent play both sides.
    #
    # Output:
    # Return a tuple:
    # - `states`: list[Board], boards seen before each move
    # - `result`: int, `1` for X win, `-1` for O win, `0` for draw
    #
    # Important:
    # Store the board before the move, not after the move.
    # That means the value table learns how good it was to be in that state.
    #
    # Steps:
    # 1. Start with `empty_board()`.
    # 2. Set `player = 1` so X moves first.
    # 3. Create an empty `states` list.
    # 4. While the game is not over:
    #    - append the current board to `states`
    #    - choose a legal move with `choose_move(...)`
    #    - apply the move
    #    - check `winner(board)`
    #    - if there is a winner, return `(states, winner_value)`
    #    - if there are no legal moves left, return `(states, 0)`
    #    - switch players with `player *= -1`
    #
    # Checks:
    # - Every stored state must be a legal Tic-Tac-Toe board.
    # - The game should never store more than 9 states.
    # - The result should always be `-1`, `0`, or `1`.
    raise NotImplementedError("Play one legal self-play Tic-Tac-Toe game.")


def update_values(values: ValueTable, states: list[Board], result: int, learning_rate: float) -> None:
    """Update the value table from one finished game."""
    # Goal:
    # Move each visited state's value toward the final result.
    # The value is always from X's point of view.
    #
    # Formula:
    # ```text
    # new_value = old_value + learning_rate * (target - old_value)
    # ```
    #
    # Meaning:
    # - `old_value` is the current value table entry for this board.
    # - `target` is the final game result from X's point of view.
    # - if X won, target is `1`
    # - if O won, target is `-1`
    # - if the game was a draw, target is `0`
    #
    # Method:
    # Use:
    # - `old_value = values.get(state, 0.0)`
    # - `values[state] = old_value + learning_rate * (result - old_value)`
    #
    # Steps:
    # 1. Loop through each board in `states`.
    # 2. Read its old value with default `0.0`.
    # 3. Apply the formula.
    # 4. Store the new value back into `values`.
    # 5. Return `None`.
    #
    # Checks:
    # - Values should stay between `-1` and `1` for this game.
    # - The function should update `values` in place.
    raise NotImplementedError("Update the Tic-Tac-Toe value table.")


def train_self_play(config: TicTacToeConfig) -> ValueTable:
    """Train a tabular value table with self-play."""
    # Goal:
    # Return a value table learned from real legal self-play games.
    #
    # Input:
    # - `config`: TicTacToeConfig
    #
    # Output:
    # - `ValueTable`
    # - Keys are boards.
    # - Values are floats from X's point of view.
    #
    # Method:
    # Use the helpers in this file:
    # - `play_self_play_game(values, config)`
    # - `update_values(values, states, result, config.learning_rate)`
    #
    # Steps:
    # 1. Set the Python random seed.
    #    Use `random.seed(config.random_state)`.
    # 2. Start with an empty value table: `values = {}`.
    # 3. Play `config.self_play_games` games.
    # 4. Update the table after each game.
    # 5. Return the value table.
    #
    # Checks:
    # - The table should contain more than one board after training.
    # - This function should use self-play games, not made-up random boards.
    raise NotImplementedError("Train a Tic-Tac-Toe value table.")


def evaluate_against_random(values: ValueTable, config: TicTacToeConfig) -> dict[str, float]:
    """Evaluate X against a random O player."""
    # Goal:
    # Measure how often the learned X player wins, draws, or loses against random O.
    #
    # Input:
    # - `values`: ValueTable learned from self-play
    # - `config`: TicTacToeConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"win_rate"`: float
    # - `"draw_rate"`: float
    # - `"loss_rate"`: float
    #
    # Evaluation setup:
    # - X uses the learned value table with no exploration.
    # - O uses a random legal move.
    # - X always moves first.
    #
    # Method:
    # Use:
    # - `choose_move(board, 1, values, epsilon=0.0)` for X
    # - `random.choice(legal_moves(board))` for O
    #
    # Steps:
    # 1. Set counters for wins, draws, and losses to `0`.
    # 2. Repeat for `config.eval_games`.
    # 3. Start each game from `empty_board()`.
    # 4. Alternate X and O moves.
    # 5. Stop on win or draw.
    # 6. Update the correct counter.
    # 7. Divide each counter by `config.eval_games`.
    # 8. Return the rate dictionary.
    #
    # Checks:
    # - The three rates should add up to about `1.0`.
    # - Evaluation should not update the value table.
    # - O should only choose legal moves.
    raise NotImplementedError("Evaluate Tic-Tac-Toe against a random player.")


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the exercise scaffold."""
    config = TicTacToeConfig()
    completed_steps = 0
    total_steps = 2

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    values, ok = report_task_progress("train_self_play", train_self_play, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("evaluate_against_random", evaluate_against_random, values, config)
    if not count_or_stop(ok):
        return
    print(metrics)


if __name__ == "__main__":
    main()
