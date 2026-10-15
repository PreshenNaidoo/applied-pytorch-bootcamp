"""
============================================================
EXERCISE 22: SMALL GO SELF PLAY MCTS
============================================================

WHAT YOU ARE BUILDING
You are trying the game-search ideas on a small Go board.

Start with 5x5 Go.
This is not full 19x19 Go.

The data comes from legal small-board Go positions.

Board shape:
- `[board_size, board_size]`
- for the default config, this is 5 rows by 5 columns

Board values:
- `0`: empty point
- `1`: black stone
- `-1`: white stone

Move values:
- `(row, col)`: place a stone on that empty point
- `None`: pass your turn

Simplified rules for this exercise:
- Captures are included.
- Suicide moves are rejected.
- Ko is not included. Ko is the rule that stops a board from repeating.
- A rollout can stop after two passes in a row or after `config.max_rollout_moves`.

This is a teaching version of Go search.
It is small enough to implement without a full Go engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random


Board = tuple[tuple[int, ...], ...]
Move = tuple[int, int] | None


@dataclass
class SmallGoConfig:
    """Settings for small Go."""

    board_size: int = 5
    mcts_simulations: int = 100
    exploration_c: float = 1.4
    komi: float = 2.5
    max_rollout_moves: int = 100
    random_state: int = 42


@dataclass
class SmallGoMCTSNode:
    """One board state in the small Go search tree."""

    board: Board
    player: int
    parent: "SmallGoMCTSNode | None" = None
    move_from_parent: Move = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[Move, "SmallGoMCTSNode"] = field(default_factory=dict)
    untried_moves: list[Move] = field(default_factory=list)


def empty_board(config: SmallGoConfig) -> Board:
    """Return an empty Go board."""
    # Goal:
    # Return a legal starting board with no stones on it.
    #
    # Input:
    # - `config.board_size`: board width and height.
    # - The default is `5`, so the board is 5 by 5.
    #
    # Output:
    # - `Board`
    # - Shape is `[board_size, board_size]` as nested tuples.
    # - Every value is `0`.
    #
    # Method:
    # Use:
    # `tuple(tuple(0 for _ in range(config.board_size)) for _ in range(config.board_size))`
    #
    # Steps:
    # 1. Make `config.board_size` rows.
    # 2. Put `config.board_size` zeros in each row.
    # 3. Return the nested tuples.
    #
    # Checks:
    # - For 5x5, there should be 25 empty points.
    # - `len(board)` should be `5`.
    # - `len(board[0])` should be `5`.
    raise NotImplementedError("Return an empty small Go board.")


def neighbors(point: tuple[int, int], config: SmallGoConfig) -> list[tuple[int, int]]:
    """Return board neighbors for one point."""
    # Goal:
    # Return up, down, left, and right neighbors that stay on the board.
    #
    # Input:
    # - `point`: `(row, col)`.
    # - For example, `(0, 0)` is the top-left corner.
    # - `config.board_size`: board width and height.
    #
    # Output:
    # - `list[tuple[int, int]]`
    # - Each item is a valid `(row, col)` point.
    #
    # Small examples on a 5x5 board:
    # - `(0, 0)` returns `[(1, 0), (0, 1)]` in any order.
    # - `(2, 2)` returns four neighbors.
    #
    # Method:
    # Try these four candidates:
    # - `(row - 1, col)`
    # - `(row + 1, col)`
    # - `(row, col - 1)`
    # - `(row, col + 1)`
    #
    # Keep a candidate only if:
    # - `0 <= candidate_row < config.board_size`
    # - `0 <= candidate_col < config.board_size`
    #
    # Steps:
    # 1. Split `point` into `row` and `col`.
    # 2. Build the four candidate points.
    # 3. Keep only candidates inside the board.
    # 4. Return the list.
    #
    # Checks:
    # - Corner `(0, 0)` on 5x5 has two neighbors.
    # - Center `(2, 2)` on 5x5 has four neighbors.
    # - No returned row or column should be negative.
    raise NotImplementedError("Return small Go point neighbors.")


def group_and_liberties(board: Board, point: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return one stone group and its liberties."""
    # Goal:
    # Find connected stones of the same color and their empty neighbor points.
    #
    # Input:
    # - `board`: legal Go board.
    # - `point`: location of one stone, not an empty point.
    # - `board[row][col]` at this point must be `1` or `-1`.
    #
    # Output:
    # - `group`: set of connected same-color points.
    # - `liberties`: set of empty neighbor points next to that group.
    #
    # Meaning:
    # - A group is stones of the same color connected up, down, left, or right.
    # - A liberty is an empty point next to that group.
    #
    # Small example:
    # On a 5x5 board with one black stone at `(2, 2)`:
    # - `group == {(2, 2)}`
    # - `liberties == {(1, 2), (3, 2), (2, 1), (2, 3)}`
    #
    # Method:
    # Use a stack or queue to visit connected same-color stones.
    # You can get the board size with `len(board)`.
    #
    # Use:
    # - `color = board[row][col]`
    # - `stack = [point]`
    # - `group = set()`
    # - `liberties = set()`
    #
    # Steps:
    # 1. Check that `point` is inside the board.
    # 2. Check that `board[row][col]` is not `0`.
    # 3. Start a stack with `point`.
    # 4. Pop one point at a time.
    # 5. Add same-color neighbors to the stack.
    # 6. Add empty neighbors to `liberties`.
    # 7. Return `(group, liberties)`.
    #
    # Checks:
    # - A single center stone on a 5x5 board has four liberties.
    # - Two connected stones should both appear in `group`.
    # - Raise `ValueError` if `point` is empty or outside the board.
    raise NotImplementedError("Return a Go group and its liberties.")


def apply_move(board: Board, move: Move, player: int, config: SmallGoConfig) -> Board:
    """Apply one legal Go move, including captures."""
    # Goal:
    # Place one stone or pass.
    # Remove opponent groups with no liberties.
    # Return a new board and leave the old board unchanged.
    #
    # Input:
    # - `board`: legal small Go board.
    # - `move`: `(row, col)` or `None`.
    # - `move=None` means pass.
    # - `player`: `1` for black, `-1` for white.
    # - `config.board_size`: board width and height.
    #
    # Output:
    # - `Board`
    # - New board after the move.
    #
    # Small examples:
    # - On an empty board, `apply_move(board, (2, 2), 1, config)`
    #   should put `1` at row `2`, column `2`.
    # - `apply_move(board, None, 1, config)` should return the same board.
    #
    # Capture rule:
    # After placing the stone, check each neighboring opponent group.
    # If an opponent group has zero liberties, remove every stone in that group.
    #
    # Suicide rule:
    # After captures, check the new stone's own group.
    # If that group has zero liberties, the move is illegal and should raise `ValueError`.
    #
    # Method:
    # Use:
    # - `rows = [list(row) for row in board]`
    # - write `player` at `rows[row][col]`
    # - convert back with `tuple(tuple(row) for row in rows)`
    #
    # Steps:
    # 1. Check that `player` is `1` or `-1`.
    # 2. If `move is None`, return `board`.
    # 3. Check that the move is inside the board.
    # 4. Check that the point is empty.
    # 5. Place the stone on a copied board.
    # 6. For each neighboring opponent group, remove it if it has no liberties.
    # 7. Check the placed stone's own group.
    # 8. If the own group has no liberties, raise `ValueError`.
    # 9. Return the new board.
    #
    # Checks:
    # - Captured stones should become `0`.
    # - Suicide moves should be rejected.
    # - Passing should not change the board.
    # - The input board should not be changed in place.
    raise NotImplementedError("Apply one small Go move.")


def legal_moves(board: Board, player: int, config: SmallGoConfig) -> list[Move]:
    """Return legal Go moves plus pass."""
    # Goal:
    # Return legal empty points and `None` for pass.
    #
    # Input:
    # - `board`: legal Go board.
    # - `player`: `1` for black or `-1` for white.
    #
    # Output:
    # - `list[Move]`
    # - A board move is `(row, col)`.
    # - `None` means pass.
    #
    # Small example:
    # On an empty 5x5 board:
    # - there are 25 points
    # - plus one pass move
    # - so the result has 26 moves
    #
    # Method:
    # Try each empty point with:
    # `apply_move(board, (row, col), player, config)`
    #
    # Keep the move if `apply_move` does not raise `ValueError`.
    # Always include `None` for pass.
    #
    # Steps:
    # 1. Start with `moves = [None]`.
    # 2. Loop over every row and column.
    # 3. Skip points that are not empty.
    # 4. Try the move with `apply_move`.
    # 5. If it works, append `(row, col)`.
    # 6. Return `moves`.
    #
    # Checks:
    # - On an empty 5x5 board, there should be 26 moves including pass.
    # - A point that already has a stone should not be returned.
    # - Suicide moves should not be returned.
    raise NotImplementedError("Return legal small Go moves.")


def area_score(board: Board, config: SmallGoConfig) -> float:
    """Return a simple area score from black's point of view."""
    # Goal:
    # Score a finished small Go board in a simple way.
    # This teaching version counts stones only.
    # It does not count surrounded empty territory.
    #
    # Input:
    # - `board`: legal Go board.
    # - `config.komi`: extra points for white because black moves first.
    #
    # Output:
    # - `float`
    # - Positive means black is ahead.
    # - Negative means white is ahead.
    #
    # Formula:
    # ```text
    # score = black_stones - white_stones - komi
    # ```
    #
    # Meaning:
    # - `black_stones` is the number of `1` values on the board.
    # - `white_stones` is the number of `-1` values on the board.
    # - `komi` gives white extra points because black moves first.
    #
    # Small example:
    # If black has 10 stones, white has 8 stones, and `komi == 2.5`,
    # then `score = 10 - 8 - 2.5 = -0.5`, so white is ahead by 0.5.
    #
    # Steps:
    # 1. Count cells equal to `1`.
    # 2. Count cells equal to `-1`.
    # 3. Return `float(black_stones - white_stones - config.komi)`.
    #
    # Checks:
    # - Empty board score should be `-config.komi`.
    # - The return value should be a Python float.
    raise NotImplementedError("Score a small Go board.")


def ucb_score(parent_visits: int, child: SmallGoMCTSNode, config: SmallGoConfig) -> float:
    """Return the UCB score for one child node."""
    # Goal:
    # Choose between moves that look good and moves that need more search.
    #
    # Input:
    # - `parent_visits`: number of times the parent node has been visited.
    # - `child`: one child node.
    # - `config.exploration_c`: exploration strength, usually around `1.4`.
    #
    # Output:
    # - `float`
    #
    # Formula:
    # ```text
    # mean_value = child.value_sum / child.visits
    # exploration = config.exploration_c * sqrt(log(parent_visits) / child.visits)
    # ucb = mean_value + exploration
    # ```
    #
    # Meaning:
    # - `mean_value` is how good this child has looked so far.
    # - `exploration` is larger for children with fewer visits.
    # - `exploration_c` controls how much the search tries less-tested moves.
    #
    # Method:
    # Use `math.log(...)` and `math.sqrt(...)`.
    # If `child.visits == 0`, return `float("inf")`.
    #
    # Checks:
    # - Unvisited children should get `float("inf")`.
    # - The score should be a Python float.
    raise NotImplementedError("Compute one small Go UCB score.")


def rollout(board: Board, player: int, config: SmallGoConfig) -> float:
    """Play random legal moves and return the final score sign."""
    # Goal:
    # Estimate a board by finishing the game with random legal moves.
    #
    # Input:
    # - `board`: legal small Go board.
    # - `player`: player to move next, either `1` or `-1`.
    # - `config.max_rollout_moves`: safety limit so rollout cannot run forever.
    #
    # Output:
    # - `float` from black's point of view:
    #   - `1.0` means black won
    #   - `-1.0` means white won
    #
    # Method:
    # Use:
    # - `moves = legal_moves(board, player, config)`
    # - `move = random.choice(moves)`
    # - `board = apply_move(board, move, player, config)`
    # - `player *= -1`
    #
    # Game-end rule:
    # Stop after two passes in a row.
    # Also stop after `config.max_rollout_moves` moves as a safety limit.
    #
    # Score rule:
    # Use `score = area_score(board, config)`.
    # Return `1.0` if `score > 0`, else return `-1.0`.
    #
    # Steps:
    # 1. Set `passes_in_a_row = 0`.
    # 2. Loop up to `config.max_rollout_moves`.
    # 3. Choose one random legal move.
    # 4. If the move is `None`, add one to `passes_in_a_row`.
    # 5. If the move is not `None`, apply it and reset `passes_in_a_row = 0`.
    # 6. Stop if `passes_in_a_row == 2`.
    # 7. Switch players after each turn.
    # 8. Score the final board and return the winner sign.
    #
    # Checks:
    # - The rollout should only use moves from `legal_moves`.
    # - The return value should be `1.0` or `-1.0`.
    raise NotImplementedError("Run one small Go rollout.")


def mcts_move(board: Board, player: int, config: SmallGoConfig) -> Move:
    """Choose a move with MCTS."""
    # Goal:
    # Search legal Go moves and return one move.
    #
    # Input:
    # - `board`: legal small Go board.
    # - `player`: player to move next, either `1` or `-1`.
    # - `config.mcts_simulations`: number of simulations to run.
    #
    # Output:
    # - `Move`
    # - one legal move, either `(row, col)` or `None` for pass.
    #
    # Method:
    # Build the root node:
    # ```python
    # root = SmallGoMCTSNode(board=board, player=player)
    # root.untried_moves = legal_moves(board, player, config)
    # ```
    #
    # One MCTS simulation has four phases:
    # 1. Selection:
    #    While the node has no untried moves and has children,
    #    move to the child with the largest UCB score.
    # 2. Expansion:
    #    If the node has untried moves, pop one move and create a child.
    # 3. Rollout:
    #    Call `rollout(node.board, node.player, config)`.
    # 4. Backpropagation:
    #    Add one visit and the rollout result to every node on the path.
    #
    # Expansion call shape:
    # ```python
    # move = node.untried_moves.pop()
    # child_board = apply_move(node.board, move, node.player, config)
    # child = SmallGoMCTSNode(
    #     board=child_board,
    #     player=-node.player,
    #     parent=node,
    #     move_from_parent=move,
    # )
    # child.untried_moves = legal_moves(child_board, child.player, config)
    # node.children[move] = child
    # ```
    #
    # Value convention:
    # - `rollout` returns from black's point of view.
    # - Store that same value in `value_sum`.
    # - This is a simple teaching version.
    #
    # Final move:
    # After all simulations, return the root child with the largest `visits`.
    #
    # Steps:
    # 1. Set `random.seed(config.random_state)` for repeatable random rollouts.
    # 2. Get legal root moves.
    # 3. If there is only one legal move, return it.
    # 4. Build the root node.
    # 5. Run `config.mcts_simulations` simulations.
    # 6. For each simulation, do selection, expansion, rollout, and backpropagation.
    # 7. Return the most visited root move.
    #
    # Checks:
    # - The returned move must be in `legal_moves(board, player, config)`.
    # - Root visits should equal `config.mcts_simulations` after the loop.
    # - If the best move is pass, returning `None` is allowed.
    raise NotImplementedError("Choose a small Go move with MCTS.")


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
    config = SmallGoConfig()
    board, ok = report_task_progress("empty_board", empty_board, config)
    if ok:
        move, move_ok = report_task_progress("mcts_move", mcts_move, board, 1, config)
        if move_ok:
            print(move)
