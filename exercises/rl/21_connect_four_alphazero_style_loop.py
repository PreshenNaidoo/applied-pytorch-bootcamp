"""
============================================================
EXERCISE 21: CONNECT FOUR ALPHAZERO STYLE LOOP
============================================================

WHAT YOU ARE BUILDING
You are building a small AlphaZero-style loop for Connect Four.

The loop uses real game states from legal self-play.
MCTS improves the move target, then the network learns from that target.

Plain-language idea:
- The network first guesses which moves look good.
- MCTS searches from the real board and counts which moves it visits most.
- The visit counts become a better policy target.
- The value target is the final game result from the player-to-move point of view.

Board shape:
- 6 rows by 7 columns

Encoded board shape:
- `[2, 6, 7]`
- plane 0 marks player one pieces
- plane 1 marks player two pieces

Training target shapes:
- policy target: tensor shaped `[7]`, one probability for each column
- value target: one float shaped `[1]`

Small policy target example:
- If columns 0, 3, and 4 are legal, and MCTS visits them `[10, 30, 10]`,
  the target over all 7 columns is `[0.2, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0]`.

Value target meanings from the stored player's point of view:
- `1.0`: the stored player won
- `0.0`: draw
- `-1.0`: the stored player lost
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


Board = tuple[tuple[int, ...], ...]


@dataclass
class AlphaZeroConnectFourConfig:
    """Settings for the tiny AlphaZero-style loop."""

    self_play_games: int = 100
    mcts_simulations: int = 50
    batch_size: int = 64
    training_epochs: int = 3
    learning_rate: float = 1e-3
    random_state: int = 42


def mcts_policy_target(board: Board, player: int, model: nn.Module, config: AlphaZeroConnectFourConfig) -> torch.Tensor:
    """Return an improved policy target from MCTS visit counts."""
    # Goal:
    # Use MCTS to make a better move target than the raw network policy.
    # The target says how much search liked each column.
    #
    # Input:
    # - `board`: legal Connect Four board with 6 rows and 7 columns.
    # - `player`: player to move next, either `1` or `-1`.
    # - `model`: policy/value network from the related exercise.
    #   It should return `(policy_logits, value)` from an encoded board batch.
    # - `config.mcts_simulations`: number of search simulations to run.
    #
    # Output:
    # - `torch.Tensor` shaped `[7]`, dtype `torch.float32`
    # - Each entry is the probability target for that column.
    # - Illegal columns must have probability `0.0`.
    # - Legal columns should sum to `1.0`.
    #
    # Small example:
    # If root child visits are:
    # - column 0: 10 visits
    # - column 3: 30 visits
    # - column 4: 10 visits
    # then return:
    # `torch.tensor([0.2, 0.0, 0.0, 0.6, 0.2, 0.0, 0.0])`
    #
    # Method:
    # Reuse the MCTS structure from the related exercise, but keep the root node
    # so you can read each child's visit count.
    #
    # Important:
    # Do not call a helper that only returns one best move.
    # This function needs all root child visit counts, not just one column.
    #
    # Use the model like this during search:
    # ```python
    # model.eval()
    # with torch.no_grad():
    #     encoded = encode_board(board).unsqueeze(0)
    #     policy_logits, value = model(encoded)
    #     policy_probs = torch.softmax(policy_logits[0], dim=-1)
    # ```
    #
    # Meaning:
    # - `encode_board(board)` should return shape `[2, 6, 7]`.
    # - `.unsqueeze(0)` adds the batch dimension, so the shape is `[1, 2, 6, 7]`.
    # - `torch.softmax(..., dim=-1)` turns 7 logits into 7 probabilities.
    #
    # A simple teaching version is fine:
    # 1. Use model policy probabilities to prefer expansion or rollout moves.
    # 2. Use the model value instead of a random rollout value.
    # 3. After all simulations, normalize root child visits.
    #
    # Visit-count formula:
    # ```text
    # target[column] = visits_for_column / total_root_child_visits
    # ```
    #
    # If `total_root_child_visits == 0`, fall back to a uniform distribution
    # over legal moves. For example, 7 legal moves gives each legal move `1 / 7`.
    #
    # Steps:
    # 1. Get the legal moves for the root board.
    # 2. If there is only one legal move, return a one-hot target for that column.
    # 3. Build the root MCTS node.
    # 4. Run `config.mcts_simulations` simulations.
    # 5. Create `target = torch.zeros(7, dtype=torch.float32)`.
    # 6. For each root child, write its visit count into `target[move]`.
    # 7. Divide `target` by `target.sum()` so the probabilities sum to `1.0`.
    # 8. Return the target.
    #
    # Checks:
    # - `target.shape` should be `(7,)`.
    # - `target.dtype` should be `torch.float32`.
    # - Illegal moves should have probability `0.0`.
    # - `abs(float(target.sum()) - 1.0)` should be small when moves are available.
    raise NotImplementedError("Build one MCTS policy target.")


def self_play_with_mcts(model: nn.Module, config: AlphaZeroConnectFourConfig) -> list[tuple[Board, torch.Tensor, float]]:
    """Generate one self-play game using MCTS-guided moves."""
    # Goal:
    # Produce real training examples from legal Connect Four self-play.
    # Store the board, the MCTS policy target, and the final result.
    #
    # Input:
    # - `model`: current policy/value network.
    # - `config`: self-play and MCTS settings.
    #
    # Output:
    # - `list[tuple[Board, torch.Tensor, float]]`
    # - Each item is `(board, improved_policy, outcome)`.
    # - `board` is the board before the stored player moved.
    # - `improved_policy` is shaped `[7]`.
    # - `outcome` is the final result from that stored player's point of view.
    #
    # Outcome formula:
    # ```text
    # outcome = final_winner * stored_player
    # ```
    #
    # Meaning:
    # - `final_winner` is `1` if player one won, `-1` if player two won, or `0` for draw.
    # - `stored_player` is the player who was about to move on the stored board.
    # - If player two was about to move and player two won, `-1 * -1 = 1`,
    #   so the outcome for that stored player is `1.0`.
    #
    # Method:
    # Use:
    # - `policy = mcts_policy_target(board, player, model, config)`
    # - `move = int(torch.multinomial(policy, num_samples=1).item())`
    # - `board = apply_move(board, move, player)`
    # - `player *= -1`
    #
    # `torch.multinomial(policy, num_samples=1)` samples one column using
    # the probabilities in `policy`. If you want a deterministic teaching
    # version, use `move = int(policy.argmax().item())` instead.
    #
    # Steps:
    # 1. Start from an empty board.
    # 2. Set `player = 1`.
    # 3. Create an empty list for `(board, player, policy)` items.
    # 4. While the game is not over:
    #    - get `policy = mcts_policy_target(board, player, model, config)`
    #    - store `(board, player, policy)`
    #    - choose a legal move from the policy
    #    - apply the move
    #    - check for a winner
    #    - if the board is full, end with draw
    #    - switch players
    # 5. Convert each stored `(board, player, policy)` into
    #    `(board, policy, float(final_winner * player))`.
    # 6. Return the converted list.
    #
    # Checks:
    # - Every board should be legal.
    # - Every improved policy should be shaped `(7,)`.
    # - Every improved policy should sum to `1.0`.
    # - Every sampled move should be legal for its board.
    # - A game should store at most 42 examples.
    raise NotImplementedError("Generate one MCTS-guided self-play game.")


def build_replay_loader(
    examples: list[tuple[Board, torch.Tensor, float]],
    config: AlphaZeroConnectFourConfig,
) -> DataLoader:
    """Build a dataloader from self-play examples."""
    # Goal:
    # Turn stored self-play examples into training batches.
    #
    # Input:
    # - `examples`: list from `self_play_with_mcts(model, config)`.
    # - Each example is `(board, improved_policy, outcome)`.
    #
    # Small input example:
    # ```python
    # (
    #     empty_board,
    #     torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    #     1.0,
    # )
    # ```
    #
    # Output:
    # - `DataLoader`
    # - Each batch should return:
    #   - `boards`: float tensor shaped `[batch_size, 2, 6, 7]`
    #   - `policy_targets`: float tensor shaped `[batch_size, 7]`
    #   - `value_targets`: float tensor shaped `[batch_size, 1]`
    #
    # Method:
    # Use:
    # - `encoded_boards = [encode_board(board) for board, _, _ in examples]`
    # - `boards = torch.stack(encoded_boards)`
    # - `policy_targets = torch.stack([policy for _, policy, _ in examples])`
    # - `value_targets = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)`
    # - `dataset = TensorDataset(boards, policy_targets, value_targets)`
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Meaning:
    # - `torch.stack` joins many `[2, 6, 7]` board tensors into `[num_examples, 2, 6, 7]`.
    # - `.unsqueeze(1)` changes value targets from `[num_examples]` to `[num_examples, 1]`.
    #
    # Steps:
    # 1. Check that `examples` is not empty.
    # 2. Encode every board into shape `[2, 6, 7]`.
    # 3. Stack encoded boards.
    # 4. Stack policy targets.
    # 5. Convert outcomes into a float tensor shaped `[num_examples, 1]`.
    # 6. Build a `TensorDataset`.
    # 7. Return a shuffled `DataLoader`.
    #
    # Checks:
    # - No example should come from an illegal board.
    # - `boards.shape[1:]` should be `(2, 6, 7)`.
    # - `policy_targets.shape[1]` should be `7`.
    # - Each policy target should sum to about `1.0`.
    # - `value_targets.shape[1]` should be `1`.
    raise NotImplementedError("Build the AlphaZero-style replay dataloader.")


def alphazero_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    """Return AlphaZero-style policy plus value loss."""
    # Goal:
    # Train the network to copy the MCTS policy and predict the game outcome.
    #
    # Input:
    # - `policy_logits`: tensor shaped `[batch_size, 7]`
    # - `values`: tensor shaped `[batch_size, 1]`
    # - `policy_targets`: tensor shaped `[batch_size, 7]`
    # - `value_targets`: tensor shaped `[batch_size, 1]`
    #
    # Output:
    # - scalar tensor shaped `[]`
    #
    # Formula:
    # ```text
    # log_probs = log_softmax(policy_logits, dim=-1)
    # policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
    # value_loss = mse_loss(values, value_targets)
    # total_loss = policy_loss + value_loss
    # ```
    #
    # Meaning:
    # - `policy_targets` is a full probability distribution, not one class id.
    # - The policy loss gives a bigger penalty when the network gives low
    #   probability to moves that MCTS visited often.
    # - The value loss trains the value head to predict win, draw, or loss.
    #
    # Method:
    # Use:
    # - `log_probs = nn.functional.log_softmax(policy_logits, dim=-1)`
    # - `policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()`
    # - `value_loss = nn.functional.mse_loss(values, value_targets)`
    #
    # Do not use:
    # - `nn.functional.cross_entropy(policy_logits, policy_targets)`
    #
    # `cross_entropy` is for class ids shaped `[batch_size]`.
    # Here the target is a soft target shaped `[batch_size, 7]`.
    #
    # Steps:
    # 1. Convert policy logits into log probabilities.
    # 2. Multiply log probabilities by the MCTS policy targets.
    # 3. Sum over the 7 columns.
    # 4. Negate and average over the batch.
    # 5. Compute MSE value loss.
    # 6. Return policy loss plus value loss.
    #
    # Checks:
    # - `policy_targets` should be probabilities, not class ids.
    # - Each row of `policy_targets` should sum to about `1.0`.
    # - The returned loss should be one scalar tensor.
    raise NotImplementedError("Compute AlphaZero-style loss.")


def run_alphazero_style_loop(config: AlphaZeroConnectFourConfig) -> dict[str, float]:
    """Run one tiny AlphaZero-style training loop."""
    # Goal:
    # Generate self-play data, train the model, and return a small summary.
    #
    # Input:
    # - `config`: settings for self-play, MCTS, and training.
    #
    # Output:
    # Return this exact dictionary:
    # - `"examples"`: number of training examples, as a float
    # - `"final_loss"`: final training loss, as a float
    #
    # You may also include:
    # - `"mean_game_length"`: average number of stored boards per game, as a float
    #
    # Method:
    # Build the policy/value network from the related exercise.
    # If you keep this file standalone, copy the same `ConnectFourPolicyValueNet`
    # class and `encode_board` helper into this file.
    #
    # Use:
    # - `torch.manual_seed(config.random_state)`
    # - `model = ConnectFourPolicyValueNet(policy_value_config)`
    # - `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    #
    # Generate data:
    # ```python
    # all_examples = []
    # game_lengths = []
    # for _ in range(config.self_play_games):
    #     game_examples = self_play_with_mcts(model, config)
    #     all_examples.extend(game_examples)
    #     game_lengths.append(len(game_examples))
    # ```
    #
    # Train on each batch:
    # ```python
    # policy_logits, values = model(boards)
    # loss = alphazero_loss(policy_logits, values, policy_targets, value_targets)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # ```
    #
    # Steps:
    # 1. Build the policy/value network.
    # 2. Generate self-play examples with MCTS.
    # 3. Build a replay dataloader.
    # 4. Build the AdamW optimizer.
    # 5. Set `final_loss = 0.0`.
    # 6. Loop for `config.training_epochs`.
    # 7. Loop over batches from the replay dataloader.
    # 8. Run the model.
    # 9. Compute AlphaZero loss.
    # 10. Clear gradients, backpropagate, and step the optimizer.
    # 11. Store `float(loss.item())` as `final_loss`.
    # 12. Return the summary dictionary.
    #
    # Checks:
    # - Return keys should include `"examples"` and `"final_loss"`.
    # - `"examples"` should be greater than `0.0`.
    # - `"final_loss"` should be a finite Python float.
    # - Training should use legal self-play examples, not random board tensors.
    raise NotImplementedError("Run the Connect Four AlphaZero-style loop.")


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
    config = AlphaZeroConnectFourConfig()
    metrics, ok = report_task_progress("run_alphazero_style_loop", run_alphazero_style_loop, config)
    if ok:
        print(metrics)
