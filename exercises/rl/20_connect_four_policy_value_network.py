"""
============================================================
EXERCISE 20: CONNECT FOUR POLICY VALUE NETWORK
============================================================

WHAT YOU ARE BUILDING
You are training one network with two heads:
- a policy head for legal move choice
- a value head for win chance

The data comes from real Connect Four self-play games.
Do not train on random board tensors.

Board shape:
- 6 rows by 7 columns

Encoded board shape:
- `[2, 6, 7]`
- plane 0 marks player one pieces
- plane 1 marks player two pieces

Training target shapes:
- policy target: one legal column index from `0` to `6`
- value target: final game result from player one's point of view

Value target meanings:
- `1.0`: player one won
- `0.0`: draw
- `-1.0`: player two won
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PolicyValueConfig:
    """Settings for the Connect Four policy/value model."""

    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    self_play_games: int = 1000


class ConnectFourPolicyValueNet(nn.Module):
    """Network with policy and value heads."""

    def __init__(self, config: PolicyValueConfig) -> None:
        # Goal:
        # Build a shared trunk plus two output heads.
        # The trunk reads the board.
        # The policy head predicts which column was played.
        # The value head predicts the final game result.
        #
        # Input:
        # - board tensor shaped `[batch_size, 2, 6, 7]`
        #
        # Output:
        # - policy logits shaped `[batch_size, 7]`
        # - value shaped `[batch_size, 1]`
        #
        # Method:
        # Use `flat_size = 2 * 6 * 7`.
        #
        # Use this exact shared trunk:
        # ```python
        # self.trunk = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(flat_size, config.hidden_dim),
        #     nn.ReLU(),
        # )
        # ```
        #
        # Use these exact heads:
        # ```python
        # self.policy_head = nn.Linear(config.hidden_dim, 7)
        # self.value_head = nn.Linear(config.hidden_dim, 1)
        # ```
        #
        # Meaning:
        # - `nn.Flatten()` changes `[batch_size, 2, 6, 7]` into `[batch_size, 84]`.
        # - The policy head has 7 outputs because Connect Four has 7 columns.
        # - The value head has 1 output because each board gets one final-result estimate.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Set `flat_size = 2 * 6 * 7`.
        # 3. Build `self.trunk`.
        # 4. Build `self.policy_head`.
        # 5. Build `self.value_head`.
        #
        # Checks:
        # - A batch of 4 boards should return shapes `[4, 7]` and `[4, 1]`.
        # - Do not put softmax in the model. Cross-entropy expects raw logits.
        super().__init__()
        raise NotImplementedError("Build the Connect Four policy/value network.")

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and value."""
        # Goal:
        # Run the shared trunk and both heads.
        #
        # Input:
        # - `boards`: tensor shaped `[batch_size, 2, 6, 7]`.
        #
        # Output:
        # - `policy_logits`: tensor shaped `[batch_size, 7]`.
        # - `value`: tensor shaped `[batch_size, 1]`.
        #
        # Method:
        # Use:
        # - `hidden = self.trunk(boards)`
        # - `policy_logits = self.policy_head(hidden)`
        # - `value = self.value_head(hidden)`
        #
        # Steps:
        # 1. Pass boards through the trunk.
        # 2. Pass hidden features through the policy head.
        # 3. Pass hidden features through the value head.
        # 4. Return `(policy_logits, value)`.
        #
        # Checks:
        # - Return raw policy logits, not softmax probabilities.
        # - Return value shape `[batch_size, 1]`, not `[batch_size]`.
        raise NotImplementedError("Return policy logits and value.")


def build_policy_value_loader(config: PolicyValueConfig) -> DataLoader:
    """Build training data from legal self-play games."""
    # Goal:
    # Return real board states, move targets, and value targets from legal self-play games.
    #
    # Input:
    # - `config`: PolicyValueConfig
    #
    # Output:
    # - `DataLoader`
    # - Each batch should return three tensors:
    #   - `boards`: float tensor shaped `[batch_size, 2, 6, 7]`
    #   - `policy_targets`: long tensor shaped `[batch_size]`
    #   - `value_targets`: float tensor shaped `[batch_size, 1]`
    #
    # Target meaning:
    # - `policy_targets[i]` is the legal column that was played from `boards[i]`.
    # - `value_targets[i]` is the final game result from player one's point of view.
    #
    # Small example:
    # If a self-play game starts with player one dropping in column `3`,
    # store the empty board as one training board and store policy target `3`.
    #
    # Method:
    # Generate legal self-play Connect Four games.
    # You can reuse the board helpers from the previous Connect Four exercises:
    # - `empty_board()`
    # - `legal_moves(board)`
    # - `apply_move(board, column, player)`
    # - `winner(board)`
    # - `encode_board(board)`
    #
    # If you keep this file standalone, include small helper functions with those same behaviors.
    #
    # Use:
    # - `torch.stack(encoded_boards)` for boards
    # - `torch.tensor(policy_targets, dtype=torch.long)` for policy targets
    # - `torch.tensor(value_targets, dtype=torch.float32).unsqueeze(1)` for value targets
    # - `TensorDataset(boards, policy_targets, value_targets)`
    # - `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Steps:
    # 1. Set the Python random seed if you use random self-play.
    # 2. Create empty lists for boards, policy targets, and value targets.
    # 3. Play `config.self_play_games` legal games.
    # 4. During each game, store each board before the move and the legal column played.
    # 5. After the game ends, assign the final result to every board from that game.
    # 6. Encode each board as `[2, 6, 7]`.
    # 7. Convert all lists into tensors with the shapes above.
    # 8. Wrap them in a `TensorDataset`.
    # 9. Return a shuffled `DataLoader`.
    #
    # Checks:
    # - Every policy target should be between 0 and 6.
    # - Every policy target should be legal for its matching board.
    # - `boards.shape[1:]` should be `(2, 6, 7)`.
    # - `value_targets` should only contain `-1.0`, `0.0`, or `1.0`.
    # - Do not use random board tensors that were not produced by legal moves.
    raise NotImplementedError("Build a policy/value dataloader from self-play.")


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    """Return policy loss plus value loss."""
    # Goal:
    # Train both heads at the same time.
    # The policy head learns the move that was played.
    # The value head learns the final game result.
    #
    # Input:
    # - `policy_logits`: tensor shaped `[batch_size, 7]`
    # - `values`: tensor shaped `[batch_size, 1]`
    # - `policy_targets`: tensor shaped `[batch_size]`, dtype `torch.long`
    # - `value_targets`: tensor shaped `[batch_size, 1]`, dtype `torch.float32`
    #
    # Output:
    # - scalar tensor shaped `[]`
    #
    # Formula:
    # ```text
    # policy_loss = cross_entropy(policy_logits, policy_targets)
    # value_loss = mse_loss(values, value_targets)
    # total_loss = policy_loss + value_loss
    # ```
    #
    # Meaning:
    # - Cross-entropy trains the policy head to put high score on the played legal column.
    # - MSE trains the value head to predict the final result.
    #
    # Method:
    # Use:
    # - `nn.functional.cross_entropy(policy_logits, policy_targets)`
    # - `nn.functional.mse_loss(values, value_targets)`
    #
    # Steps:
    # 1. Compute policy loss.
    # 2. Compute value loss.
    # 3. Add them.
    # 4. Return the total loss.
    #
    # Checks:
    # - Policy logits should be `[batch_size, 7]`.
    # - Policy targets should be `[batch_size]`.
    # - Do not apply softmax before `cross_entropy`.
    # - The returned loss should be one scalar tensor.
    raise NotImplementedError("Compute Connect Four policy/value loss.")


def train_policy_value_model(config: PolicyValueConfig) -> dict[str, float]:
    """Train the policy/value network."""
    # Goal:
    # Train the policy/value network from real legal Connect Four self-play data.
    #
    # Input:
    # - `config`: PolicyValueConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"final_loss"`: float
    #
    # Method:
    # Use:
    # - `loader = build_policy_value_loader(config)`
    # - `model = ConnectFourPolicyValueNet(config)`
    # - `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    #
    # For each batch:
    # - `policy_logits, values = model(boards)`
    # - `loss = policy_value_loss(policy_logits, values, policy_targets, value_targets)`
    # - `optimizer.zero_grad()`
    # - `loss.backward()`
    # - `optimizer.step()`
    #
    # Steps:
    # 1. Build the DataLoader.
    # 2. Build the model.
    # 3. Build the optimizer.
    # 4. Set `final_loss = 0.0`.
    # 5. Loop for `config.epochs`.
    # 6. Loop over batches from the loader.
    # 7. Run the forward pass.
    # 8. Compute total loss.
    # 9. Clear gradients, backpropagate, and step the optimizer.
    # 10. Store `float(loss.item())` as `final_loss`.
    # 11. Return `{"final_loss": final_loss}`.
    #
    # Checks:
    # - Loss should be finite, not NaN.
    # - `final_loss` should be a Python float.
    # - Training should use legal self-play data from the loader.
    raise NotImplementedError("Train the Connect Four policy/value network.")


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
    config = PolicyValueConfig()
    metrics, ok = report_task_progress("train_policy_value_model", train_policy_value_model, config)
    if ok:
        print(metrics)
