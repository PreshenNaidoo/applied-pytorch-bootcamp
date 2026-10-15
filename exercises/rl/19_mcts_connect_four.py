"""
============================================================
EXERCISE 19: MCTS CONNECT FOUR
============================================================

WHAT YOU ARE BUILDING
You are adding Monte Carlo Tree Search, or MCTS, to Connect Four.

MCTS searches real legal game states.
It does not use random tensors as data.

Plain-language meanings:
- a node is one board state in the search tree
- a child is a board that comes after one legal move
- selection means walking down the tree using UCB scores
- expansion means adding a new child for a legal move not tried yet
- rollout means finishing the game with random legal moves
- backpropagation means sending the final game result back up the tree

The result is always from player one's point of view:
- `1` means player one won
- `0` means draw
- `-1` means player two won
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random


Board = tuple[tuple[int, ...], ...]


@dataclass
class MCTSConfig:
    """Settings for Connect Four MCTS."""

    simulations: int = 200
    exploration_c: float = 1.4
    random_state: int = 42


@dataclass
class MCTSNode:
    """One node in the search tree."""

    board: Board
    player: int
    parent: "MCTSNode | None" = None
    move_from_parent: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)


def legal_moves(board: Board) -> list[int]:
    """Return legal Connect Four columns."""
    # Goal:
    # Return columns that still have room for a piece.
    #
    # Input:
    # - `board`: legal Connect Four board shaped 6 rows by 7 columns
    # - row `0` is the top row
    # - row `5` is the bottom row
    #
    # Output:
    # - `list[int]`
    # - Legal column indexes from `0` to `6`.
    #
    # Method:
    # A column is legal when its top cell is `0`.
    # The top cell is `board[0][column]`.
    #
    # Steps:
    # 1. Create an empty `moves` list.
    # 2. Loop over columns `0` through `6`.
    # 3. If `board[0][column] == 0`, add the column to `moves`.
    # 4. Return `moves`.
    #
    # Checks:
    # - Empty board should return 7 moves.
    # - A full column should not appear in the result.
    raise NotImplementedError("Return legal Connect Four columns.")


def apply_move(board: Board, column: int, player: int) -> Board:
    """Apply one legal Connect Four move."""
    # Goal:
    # Drop a piece into the chosen column using real game rules.
    # Return a new board. Do not change the old board.
    #
    # Input:
    # - `board`: legal Connect Four board shaped 6 by 7
    # - `column`: integer from `0` to `6`
    # - `player`: `1` or `-1`
    #
    # Output:
    # - `Board`
    # - New legal board after gravity places the piece.
    #
    # Example:
    # On an empty board, `apply_move(board, 3, 1)` should put `1`
    # at row `5`, column `3`.
    #
    # Method:
    # Use:
    # - `rows = [list(row) for row in board]`
    # - loop over rows from `5` down to `0`
    # - put `player` in the first empty cell in the chosen column
    # - convert back with `tuple(tuple(row) for row in rows)`
    #
    # Steps:
    # 1. Check that `player` is `1` or `-1`.
    # 2. Check that `column` is in `legal_moves(board)`.
    # 3. Copy the board to a list of lists.
    # 4. Search rows from bottom row `5` up to top row `0`.
    # 5. Put the piece in the first empty cell.
    # 6. Return the new board.
    #
    # Checks:
    # - The piece should land in the lowest empty row.
    # - A second piece in the same column should land one row above the first.
    # - Raise `ValueError` for a full column or invalid player.
    raise NotImplementedError("Apply one Connect Four move.")


def winner(board: Board) -> int:
    """Return winner, or 0 when nobody has won."""
    # Goal:
    # Check horizontal, vertical, and diagonal four-in-a-row.
    #
    # Input:
    # - `board`: legal Connect Four board.
    #
    # Output:
    # - `1` when player 1 has four connected pieces.
    # - `-1` when player -1 has four connected pieces.
    # - `0` when nobody has won.
    #
    # Method:
    # For each non-empty cell, check four directions:
    # - right: `(row, col + offset)`
    # - down: `(row + offset, col)`
    # - down-right diagonal: `(row + offset, col + offset)`
    # - down-left diagonal: `(row + offset, col - offset)`
    #
    # Steps:
    # 1. Loop over every row and column.
    # 2. Skip cells with value `0`.
    # 3. For each direction, check whether four positions stay inside the board.
    # 4. If all four positions match the starting cell, return that player.
    # 5. Return `0` if no winner is found.
    #
    # Checks:
    # - A board with four connected pieces should return that player.
    # - Empty board should return `0`.
    raise NotImplementedError("Return the Connect Four winner.")


def ucb_score(parent_visits: int, child: MCTSNode, exploration_c: float) -> float:
    """Return the UCB score for one child."""
    # Goal:
    # Balance moves that look good with moves that need more trials.
    #
    # Formula:
    # ```text
    # mean_value = child.value_sum / child.visits
    # exploration = exploration_c * sqrt(log(parent_visits) / child.visits)
    # ucb = mean_value + exploration
    # ```
    #
    # Meaning:
    # - `mean_value` is how good this child has looked so far.
    # - `exploration` is larger for children with fewer visits.
    # - `exploration_c` controls how much the search tries less-tested moves.
    #
    # Input:
    # - `parent_visits`: int
    # - `child`: MCTSNode
    # - `exploration_c`: float, such as `1.4`
    #
    # Output:
    # - `float`
    #
    # Method:
    # Use `math.log(...)` and `math.sqrt(...)`.
    # If `child.visits == 0`, return `float("inf")` so unvisited moves get tried.
    #
    # Steps:
    # 1. If the child has zero visits, return infinity.
    # 2. Compute `mean_value`.
    # 3. Compute the exploration term.
    # 4. Return their sum.
    #
    # Checks:
    # - If `child.visits == 0`, return a very large number.
    # - The score should go down as `child.visits` grows, if value stays the same.
    raise NotImplementedError("Compute one UCB score.")


def select_child(node: MCTSNode, config: MCTSConfig) -> MCTSNode:
    """Choose the child with the best UCB score."""
    # Goal:
    # Pick the next node during the selection phase of tree search.
    #
    # Input:
    # - `node`: MCTSNode
    # - `node.children`: dictionary from move to child node
    # - `config`: MCTSConfig
    #
    # Output:
    # - `MCTSNode`
    # - One child from `node.children`.
    #
    # Method:
    # Use:
    # `max(node.children.values(), key=lambda child: ucb_score(node.visits, child, config.exploration_c))`
    #
    # Steps:
    # 1. Check that the node has at least one child.
    # 2. Compute a UCB score for each child.
    # 3. Return the child with the largest score.
    #
    # Checks:
    # - The returned child must come from `node.children`.
    # - Raise `ValueError` if `node.children` is empty.
    raise NotImplementedError("Select one MCTS child.")


def expand(node: MCTSNode) -> MCTSNode:
    """Expand one untried legal move."""
    # Goal:
    # Add one child node for one legal move that has not been tried yet.
    #
    # Input:
    # - `node`: MCTSNode
    # - `node.untried_moves`: legal moves that do not have children yet
    #
    # Output:
    # - `MCTSNode`
    # - The newly created child node.
    #
    # Method:
    # The child represents the board after `node.player` makes the move.
    # The child player should be `-node.player`, because turns alternate.
    #
    # Use:
    # - `move = node.untried_moves.pop()`
    # - `child_board = apply_move(node.board, move, node.player)`
    # - `child = MCTSNode(board=child_board, player=-node.player, parent=node, move_from_parent=move)`
    # - `child.untried_moves = legal_moves(child_board)`
    #
    # Steps:
    # 1. Pop one move from `node.untried_moves`.
    # 2. Apply it to make a child board.
    # 3. Create the child with the next player.
    # 4. Store it in `node.children`.
    # 5. Return the child.
    #
    # Checks:
    # - The child board should be one legal move away from the parent board.
    # - The move should no longer be in `node.untried_moves`.
    # - Raise `ValueError` if there are no untried moves.
    raise NotImplementedError("Expand one MCTS node.")


def rollout(board: Board, player: int) -> int:
    """Play random legal moves until the game ends."""
    # Goal:
    # Estimate a board by finishing the game with legal random moves.
    #
    # Input:
    # - `board`: legal board to start from.
    # - `player`: player to move next.
    #
    # Output:
    # - result from player one's point of view: `1`, `0`, or `-1`
    #
    # Method:
    # Use:
    # - `moves = legal_moves(board)`
    # - `move = random.choice(moves)`
    # - `board = apply_move(board, move, player)`
    # - `player *= -1` to switch turns
    #
    # Steps:
    # 1. Check whether the starting board already has a winner.
    # 2. While there is no winner:
    #    - get legal moves
    #    - if there are no legal moves, return `0` for draw
    #    - choose one random legal move
    #    - apply it
    #    - check for a winner
    #    - switch players if the game is not over
    # 3. Return the final winner.
    #
    # Checks:
    # - Rollout should stop when there is a win or draw.
    # - Rollout should never choose an illegal column.
    # - Return value should always be `-1`, `0`, or `1`.
    raise NotImplementedError("Run one Connect Four rollout.")


def backpropagate(node: MCTSNode, result: int) -> None:
    """Update visits and value sums back to the root."""
    # Goal:
    # Record the rollout result on every node in the path from leaf to root.
    #
    # Input:
    # - `node`: leaf node reached by the simulation.
    # - `result`: final game result from player one's point of view.
    #
    # Output:
    # - Return `None`; update nodes in place.
    #
    # Value convention:
    # - `value_sum` stores results from player one's point of view.
    # - If player one wins, add `1`.
    # - If player two wins, add `-1`.
    # - If draw, add `0`.
    #
    # Steps:
    # 1. Start from the leaf node.
    # 2. Add one to `node.visits`.
    # 3. Add `result` to `node.value_sum`.
    # 4. Move to `node.parent`.
    # 5. Repeat until there is no parent.
    #
    # Checks:
    # - Root visits should increase by one after one backpropagation.
    # - Every node on the selected path should get one extra visit.
    raise NotImplementedError("Backpropagate one MCTS result.")


def mcts_search(board: Board, player: int, config: MCTSConfig) -> int:
    """Return the move chosen by MCTS."""
    # Goal:
    # Run many simulations and choose the most visited legal move.
    #
    # Input:
    # - `board`: legal Connect Four board.
    # - `player`: player to move next.
    # - `config`: MCTSConfig
    #
    # Output:
    # - `int`
    # - One legal column index from the input board.
    #
    # Method:
    # Build the root node:
    # ```python
    # root = MCTSNode(board=board, player=player)
    # root.untried_moves = legal_moves(board)
    # ```
    #
    # One simulation has four phases:
    # 1. Selection:
    #    While the node has no untried moves and has children,
    #    move to `select_child(node, config)`.
    # 2. Expansion:
    #    If the node has untried moves and the board is not terminal,
    #    call `expand(node)`.
    # 3. Rollout:
    #    Finish the game randomly from the selected or expanded node.
    # 4. Backpropagation:
    #    Call `backpropagate(node, result)`.
    #
    # After all simulations:
    # - choose the root child with the highest `visits`
    # - return that child's `move_from_parent`
    #
    # Important:
    # If the input board has only one legal move, return it directly.
    #
    # Checks:
    # - The returned move must be legal in the input board.
    # - Root visits should equal `config.simulations` after all simulations.
    # - Do not create moves outside the legal columns.
    raise NotImplementedError("Run MCTS and return one Connect Four move.")
