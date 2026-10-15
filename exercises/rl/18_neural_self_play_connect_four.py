"""
============================================================
EXERCISE 18: NEURAL SELF PLAY CONNECT FOUR
============================================================

WHAT YOU ARE BUILDING
You are building a small neural model trained on real Connect Four self-play games.

The data comes from legal Connect Four moves.
Do not train on random board tensors.

Board shape:
- 6 rows by 7 columns

Board values:
- `0`: empty
- `1`: current player or player one
- `-1`: other player or player two
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


Board = tuple[tuple[int, ...], ...]


@dataclass
class ConnectFourSelfPlayConfig:
    """Settings for neural Connect Four self-play."""

    self_play_games: int = 1000
    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    random_state: int = 42


def empty_board() -> Board:
    """Return an empty Connect Four board."""
    # Goal:
    # Return a legal starting board before any player has moved.
    #
    # Output:
    # - `Board`
    # - Tuple of 6 rows.
    # - Each row is a tuple with 7 zeros.
    #
    # Board index layout:
    # - row `0` is the top row
    # - row `5` is the bottom row
    # - columns go from `0` to `6`
    #
    # Method:
    # Use:
    # `tuple(tuple(0 for _ in range(7)) for _ in range(6))`
    #
    # Checks:
    # - `len(board)` should be 6.
    # - `len(board[0])` should be 7.
    # - Every cell should be `0`.
    raise NotImplementedError("Return an empty Connect Four board.")


def legal_moves(board: Board) -> list[int]:
    """Return legal columns."""
    # Goal:
    # Return columns where a player can still drop a piece.
    #
    # Input:
    # - `board`: Board with shape 6 rows by 7 columns.
    #
    # Output:
    # - `list[int]`
    # - Legal column indexes from `0` to `6`.
    #
    # Method:
    # A column is legal when the top cell in that column is `0`.
    # The top cell is `board[0][column]`.
    #
    # Steps:
    # 1. Create an empty `moves` list.
    # 2. Loop over columns `0` through `6`.
    # 3. If `board[0][column] == 0`, add the column to `moves`.
    # 4. Return `moves`.
    #
    # Checks:
    # - An empty board should return all columns `[0, 1, 2, 3, 4, 5, 6]`.
    # - A full column should not appear in the result.
    raise NotImplementedError("Return legal Connect Four columns.")


def apply_move(board: Board, column: int, player: int) -> Board:
    """Drop one piece into a legal column."""
    # Goal:
    # Apply gravity and return a new board.
    # The old board should not be changed.
    #
    # Input:
    # - `board`: Board with shape 6 by 7
    # - `column`: int from `0` to `6`
    # - `player`: `1` for player one or `-1` for player two
    #
    # Output:
    # - `Board`
    # - New board after the piece falls to the lowest empty row in that column.
    #
    # Example:
    # On an empty board, `apply_move(board, column=3, player=1)` should place `1`
    # at row `5`, column `3`.
    #
    # Method:
    # Use:
    # - `rows = [list(row) for row in board]`
    # - loop over `row` from `5` down to `0`
    # - put `player` in the first cell where `rows[row][column] == 0`
    # - return `tuple(tuple(row) for row in rows)`
    #
    # Steps:
    # 1. Check that `player` is `1` or `-1`.
    # 2. Check that `column` is in `legal_moves(board)`.
    # 3. Copy the board into a list of lists.
    # 4. Search from bottom row `5` up to top row `0`.
    # 5. Place the piece in the first empty cell.
    # 6. Convert the rows back to tuples and return the new board.
    #
    # Checks:
    # - A move in an empty column should land on row 5.
    # - A second move in the same column should land on row 4.
    # - Raise `ValueError` if the column is full or outside `0..6`.
    # - Raise `ValueError` if `player` is not `1` or `-1`.
    raise NotImplementedError("Apply one Connect Four move.")


def winner(board: Board) -> int:
    """Return the winner, or 0 when nobody has won."""
    # Goal:
    # Detect four connected pieces in rows, columns, or diagonals.
    #
    # Input:
    # - `board`: Board with shape 6 rows by 7 columns.
    #
    # Output:
    # - `1`, `-1`, or `0`
    # - `1` means player one won.
    # - `-1` means player two won.
    # - `0` means nobody has won yet.
    #
    # Method:
    # Check every cell as a possible start.
    # From each cell, check four directions:
    # - right: `(row, col + offset)`
    # - down: `(row + offset, col)`
    # - down-right diagonal: `(row + offset, col + offset)`
    # - down-left diagonal: `(row + offset, col - offset)`
    #
    # For each direction:
    # 1. Make sure all four positions stay inside the board.
    # 2. Read the four values.
    # 3. If all four values are `1`, return `1`.
    # 4. If all four values are `-1`, return `-1`.
    # 5. If no four-in-a-row is found, return `0`.
    #
    # Checks:
    # - Four `1` values in the bottom row should return `1`.
    # - Four `-1` values in one column should return `-1`.
    # - Empty board should return `0`.
    raise NotImplementedError("Return the Connect Four winner.")


def play_self_play_game(config: ConnectFourSelfPlayConfig) -> list[tuple[Board, int]]:
    """Return one legal self-play game as board/result pairs."""
    # Goal:
    # Generate one real legal game from legal Connect Four moves.
    # This first neural data exercise can use random legal moves for both players.
    #
    # Input:
    # - `config`: ConnectFourSelfPlayConfig
    #
    # Output:
    # - `list[tuple[Board, int]]`
    # - Each item is `(board, outcome)`.
    # - `board` is a real board seen before a move.
    # - `outcome` is from player one's point of view:
    #   - `1` means player one won
    #   - `0` means draw
    #   - `-1` means player two won
    #
    # Method:
    # Use random legal moves:
    # - `moves = legal_moves(board)`
    # - `column = random.choice(moves)`
    # - `board = apply_move(board, column, player)`
    #
    # Store boards before moves in a `visited_boards` list.
    # After the game ends, pair every visited board with the final outcome.
    #
    # Steps:
    # 1. Start with `board = empty_board()`.
    # 2. Set `player = 1`.
    # 3. Create an empty `visited_boards` list.
    # 4. While the game is not over:
    #    - append the current board to `visited_boards`
    #    - choose a random legal column
    #    - apply the move
    #    - check `winner(board)`
    #    - if a player won, set `outcome` to that winner and stop
    #    - if there are no legal moves left, set `outcome = 0` and stop
    #    - switch players with `player *= -1`
    # 5. Return `[(board, outcome) for board in visited_boards]`.
    #
    # Checks:
    # - No board should contain a floating piece.
    # - A game should store at most 42 boards.
    # - Every outcome should be `-1`, `0`, or `1`.
    raise NotImplementedError("Generate one legal Connect Four self-play game.")


def encode_board(board: Board) -> torch.Tensor:
    """Encode one board for a neural network."""
    # Goal:
    # Convert a board into two binary planes for the neural network.
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[2, 6, 7]`
    # - dtype: `torch.float32`
    # - plane 0 is `1.0` where player one has pieces
    # - plane 1 is `1.0` where player two has pieces
    #
    # Example:
    # If `board[5][3] == 1`, then `encoded[0, 5, 3]` should be `1.0`.
    # If `board[4][3] == -1`, then `encoded[1, 4, 3]` should be `1.0`.
    #
    # Method:
    # Use:
    # - `encoded = torch.zeros((2, 6, 7), dtype=torch.float32)`
    # - loop over rows and columns
    # - if the board value is `1`, write to plane 0
    # - if the board value is `-1`, write to plane 1
    #
    # Steps:
    # 1. Create a zero tensor with shape `[2, 6, 7]`.
    # 2. Loop over every row and column.
    # 3. Fill the correct plane for player one and player two pieces.
    # 4. Return the tensor.
    #
    # Checks:
    # - Empty board encoding should sum to `0`.
    # - The encoded tensor should never contain `-1`.
    # - `encoded.shape` should be `(2, 6, 7)`.
    raise NotImplementedError("Encode one Connect Four board.")


class ConnectFourValueNet(nn.Module):
    """Small network that predicts a board value."""

    def __init__(self, config: ConnectFourSelfPlayConfig) -> None:
        # Goal:
        # - Build a model that maps board planes to one value.
        #
        # Input:
        # - board batch shaped `[batch_size, 2, 6, 7]`
        # - Flattened board size is `2 * 6 * 7 = 84`.
        #
        # Output:
        # - value tensor shaped `[batch_size, 1]`
        #
        # Method:
        # - Use this exact small model shape:
        #   `nn.Sequential(`
        #   `    nn.Flatten(),`
        #   `    nn.Linear(2 * 6 * 7, config.hidden_dim),`
        #   `    nn.ReLU(),`
        #   `    nn.Linear(config.hidden_dim, 1),`
        #   `)`
        # - `nn.Flatten()` changes `[batch_size, 2, 6, 7]` into `[batch_size, 84]`.
        # - The first `nn.Linear` maps board features to hidden features.
        # - The last `nn.Linear` maps hidden features to one value.
        #
        # Checks:
        # - One batch of 8 boards should return shape `[8, 1]`.
        super().__init__()
        raise NotImplementedError("Build the Connect Four value network.")

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """Return board values."""
        # Goal:
        # Return one predicted value per board.
        #
        # Input:
        # - `boards`: torch.Tensor with shape `[batch_size, 2, 6, 7]`
        #
        # Output:
        # - `torch.Tensor` with shape `[batch_size, 1]`
        #
        # Method:
        # Return:
        # `self.net(boards)`
        #
        # Checks:
        # - Do not apply sigmoid or tanh unless your loss expects it.
        # - A batch of 8 boards should return shape `[8, 1]`.
        raise NotImplementedError("Return Connect Four board values.")


def build_self_play_loader(config: ConnectFourSelfPlayConfig) -> DataLoader:
    """Build a dataloader from legal self-play games."""
    # Goal:
    # Collect board/value examples from real legal self-play games.
    #
    # Input:
    # - `config`: ConnectFourSelfPlayConfig
    #
    # Output:
    # - `DataLoader`
    # - Each batch should return:
    #   - `boards`: tensor shaped `[batch_size, 2, 6, 7]`
    #   - `targets`: tensor shaped `[batch_size, 1]`
    #
    # Target meaning:
    # - Target is the final game outcome from player one's point of view.
    # - `1.0` means player one won.
    # - `0.0` means draw.
    # - `-1.0` means player two won.
    #
    # Method:
    # Use:
    # - `examples = play_self_play_game(config)`
    # - `encoded_board = encode_board(board)`
    # - `boards_tensor = torch.stack(encoded_boards)`
    # - `targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)`
    # - `dataset = TensorDataset(boards_tensor, targets_tensor)`
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Steps:
    # 1. Create empty lists for encoded boards and targets.
    # 2. Loop for `config.self_play_games`.
    # 3. Play one legal self-play game.
    # 4. For each `(board, outcome)` pair, encode the board and store the outcome.
    # 5. Stack boards into one tensor.
    # 6. Convert targets into shape `[num_examples, 1]`.
    # 7. Build and return the DataLoader.
    #
    # Checks:
    # - The first batch should have legal shapes.
    # - The dataset should contain boards from legal games, not random tensors.
    # - Targets should only contain `-1.0`, `0.0`, or `1.0`.
    raise NotImplementedError("Build a Connect Four self-play dataloader.")


def train_value_model(config: ConnectFourSelfPlayConfig) -> dict[str, float]:
    """Train the neural value model."""
    # Goal:
    # Train a small neural value model on legal self-play board outcomes.
    #
    # Input:
    # - `config`: ConnectFourSelfPlayConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"final_loss"`: float
    #
    # Method:
    # Use:
    # - `random.seed(config.random_state)`
    # - `torch.manual_seed(config.random_state)`
    # - `loader = build_self_play_loader(config)`
    # - `model = ConnectFourValueNet(config)`
    # - `loss_fn = nn.MSELoss()`
    # - `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    #
    # Training loop:
    # 1. Loop for `config.epochs`.
    # 2. For each batch, compute `predictions = model(boards)`.
    # 3. Compute `loss = loss_fn(predictions, targets)`.
    # 4. Call `optimizer.zero_grad()`.
    # 5. Call `loss.backward()`.
    # 6. Call `optimizer.step()`.
    # 7. Track the most recent loss as `final_loss`.
    #
    # Checks:
    # - Return a dictionary with key `final_loss`.
    # - `final_loss` should be a Python float.
    # - The model should train on the DataLoader, not on random board tensors.
    raise NotImplementedError("Train the Connect Four value model.")


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


if __name__ == "__main__":
    config = ConnectFourSelfPlayConfig()
    metrics, ok = report_task_progress("train_value_model", train_value_model, config)
    if ok:
        print(metrics)
