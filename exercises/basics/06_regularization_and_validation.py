"""
============================================================
EXERCISE 06: REGULARIZATION AND VALIDATION
============================================================

WHAT YOU ARE BUILDING
You are building a training setup that pays attention to overfitting.

You will compare training and validation behavior, add regularization ideas, and learn how to stop training when validation stops improving.

WHY THIS EXERCISE EXISTS
A model can look good on training data and still do poorly on new data.

This exercise teaches you how to notice that problem and respond to it.

DATASET
Reuse simple synthetic vector classification data with a 60/20/20 train/val/test split.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

Generate the dataset with `sklearn.datasets.make_classification` and split it 60% train / 20% val / 20% test.

INPUTS AND OUTPUTS
Inputs:
- vector features
- class labels
- a small MLP with dropout
- validation tracking settings

Outputs:
- train and validation loss curves
- train and validation accuracy curves if you add them
- an early stopping decision

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small MLP for vector classification.

At a high level:
- one or two hidden layers
- ReLU activations
- dropout in a sensible place
- output head that matches the task

LOSS FUNCTION
Use one exact default setup in this exercise so you do not have to guess.

Use this default path:
- multiclass classification
- model returns raw logits with shape `[batch_size, num_classes]`
- labels are integer class ids with dtype `torch.long`
- loss is `CrossEntropyLoss`

The main learning goal here is regularization and validation behavior, not loss selection.

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, keep the model, inputs, and labels on the same device.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- validation accuracy

Also track:
- best validation loss so far
- best validation accuracy so far

TARGET RESULT RANGE
You want a result that is stable and reasonable.

More important than a single number:
- validation should improve early in training
- if overfitting starts, you should be able to notice it

COMMON MISTAKES
- only looking at training loss
- using dropout during evaluation by mistake
- fitting early stopping on training loss instead of validation loss
- saving curves but not reading them
- making weight decay so large that learning becomes weak

STEP-BY-STEP TODO
1. Build the baseline model and data pipeline.
2. Add dropout to the model.
3. Add weight decay in the optimizer.
4. Track train and validation loss across epochs.
5. Plot the curves with `common.plotting.plot_loss_curves`.
6. Write an early stopping rule.
7. Explain whether you saw overfitting.

STRETCH GOALS
- compare no dropout vs dropout
- compare no weight decay vs weight decay
- save one plot to a file in the bootcamp folder

DEFINITION OF DONE
You are done when:
- you can train and validate end to end
- you can make and read a loss curve plot
- you can explain whether the model is overfitting
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.plotting import plot_loss_curves
from common.training_checks import print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
        #raise NotImplementedError("Return raw class logits from the regularized model.")


# Build the synthetic dataset and loaders.
# Use a 60% train / 20% val / 20% test split.
# Return a dict with exactly these keys:
# - "train"
# - "val"
# - "test"
# Features should be float tensors.
# Labels should be integer class tensors with dtype `torch.long`.
def load_dataloaders(config: RegularizationConfig) -> dict[str, DataLoader]:

    x, y = make_classification(n_samples = config.num_samples, n_features=config.num_features, 
                                n_classes=config.num_classes, n_informative=config.num_features//2)

    num_train, num_val = int(config.num_samples * 0.6), int(config.num_samples * 0.2)
    
    train_ds = TensorDataset(torch.tensor(x[:num_train,:], dtype=torch.float), torch.tensor(y[:num_train], dtype=torch.int64))
    val_ds = TensorDataset(torch.tensor(x[num_train:num_train+num_val,:], dtype=torch.float), torch.tensor(y[num_train:num_train+num_val], dtype=torch.int64))
    test_ds = TensorDataset(torch.tensor(x[num_train+num_val:,:], dtype=torch.float), torch.tensor(y[num_train+num_val:], dtype=torch.int64))

    return {"train": DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": DataLoader(dataset=test_ds, batch_size=config.batch_size)}

class RegularizationConfig:
    """Settings for regularization and validation."""

    num_samples: int = 1200
    num_features: int = 20
    num_classes: int = 3
    hidden_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 3000
    patience: int = 5
    random_state: int = 42


class RegularizedMLP(nn.Module):
    """MLP stub with a place for dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2)
        self.d1 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(in_features=hidden_dim//2, out_features=num_classes)

        #raise NotImplementedError("Build a small MLP and place dropout after a hidden activation. Keep it simple.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Important:
        - return raw logits
        - dropout should run in train mode and turn off in eval mode automatically
        """

        x = self.fc1(x)
        x = F.relu(x)
        x = self.d1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # weight decay = 1e-4
        # without droput, with weight decay: Best Val Loss: 0.41208744049072266, early_stopped_at_epoch: 786, val_acc: 87.89
        # with drop_out, with weight decay: Best Val Loss: 0.47408461570739746, early_stopped_at_epoch: 400, val_acc: 82
        # with drop_out, without weight decay: Best Val Loss: 0.1973285973072052, early_stopped_at_epoch: 841, val_acc: 92.5
        # without both: Best Val Loss: 0.45213884115219116, early_stopped_at_epoch: 800, val_acc: 82.4

        # Your numbers:
        # dropout + weight decay: val loss 0.474, val acc 82
        # dropout without weight decay: val loss 0.197, val acc 92.5
        # That gap is large. That usually means:
        # the combination is over-regularizing
        # or your weight decay is too strong when dropout is also on
        # So the likely issue is:
        # dropout alone helped
        # weight decay alone helped somewhat
        # both together were too aggressive
        # What to try first:
        # Keep dropout on.
        # Lower weight decay.
        # Then test patience.

        #ok lets try weight decay 1e-5
        # Best Val Loss: 0.3061606287956238, early_stopped_at_epoch: 923, val_acc: 88.3

        # Weight decay is usually the same idea as L2 regularization, not L1 regularization.
        # What L2 does:
        # penalizes large weights
        # pushes weights to become smaller
        # usually shrinks them smoothly toward zero
        # What L1 does:
        # also penalizes weights
        # but it pushes more weights to become exactly zero
        # this can make the model more sparse

        # L2 regularization means adding a penalty based on the model weights, not based on the loss itself.
        # loss + lambda_value * sum_of_squared_weights
        # Simple example:
                # l2_penalty = 0.0
                # for param in model.parameters():
                #     l2_penalty = l2_penalty + (param ** 2).sum()

                # total_loss = loss + 1e-5 * l2_penalty

        # The idea is that large weights should cost something extra. So the total loss 
        # becomes “prediction loss plus a weight penalty.” This makes the model try to 
        # fit the data well while also keeping its weights smaller. In simple words, 
        # normal loss asks, “How wrong are the predictions?” L2 regularization adds 
        # another question: “Are the weights getting too large?” If the weights grow a 
        # lot, the penalty grows too. That makes the total loss bigger, so training will 
        # try to reduce both the prediction error and the size of the weights. This 
        # often helps the model avoid overfitting.

        # So yes:
        # the total loss becomes bigger
        # but specifically when the weights are large
        # That makes learning try to do two things:
        # reduce prediction error
        # reduce large weight values
        # So the model does not just “work harder to reduce loss”
        # It works to reduce a new combined objective:
            #total_loss = data_loss + regularization_penalty
        # Very plain version:
        # normal loss cares about predictions
        # L2 adds a cost for large weights
        # so the model learns to fit the data without letting weights grow too much

        # L1 and L2 regularization both try to stop the model from relying on very large weights, 
        # but they do it in different ways. L1 regularization tends to push some weights all the 
        # way to zero, so it can make the model use fewer features. L2 regularization usually does 
        # not make weights exactly zero, but it pushes them to become smaller and smoother overall. 
        # In simple words, L1 often acts more like “remove some weights,” while L2 acts more 
        # like “shrink all weights.”

        # L1 gives a penalty based on the absolute value of each weight. Because of that, it tends 
        # to push some weights all the way to zero. L2 gives a penalty based on the square of each 
        # weight, so it punishes very large weights more strongly, but usually shrinks them smoothly 
        # instead of making them exactly zero.

        # If the weight is less than 1 in size, L1 gives a bigger penalty than L2. That is because:
        # L1 uses the absolute value
        # L2 uses the squared value
        # So for a weight like 0.2:
        # L1 gives 0.2
        # L2 gives 0.04
        # That means L1 pushes small weights harder than L2 does. This is one reason L1 often drives small weights all the way to zero.
        # But once weights get bigger than 1 in size, L2 becomes larger very quickly because of the square. So:
        # for small weights, L1 penalizes more
        # for large weights, L2 penalizes more


        return x


    #raise NotImplementedError("Return train, val, and test loaders.")


# Build the three training objects.
# Return:
# - model
# - loss_fn
# - optimizer
# Use CrossEntropyLoss for the loss.
# Put config.weight_decay into the optimizer.
# Important:
# - the model returns raw logits
# - the labels are integer class ids with dtype `torch.long`
def build_training_objects(config: RegularizationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(config.random_state)

    model = RegularizedMLP(input_dim = config.num_features, hidden_dim=config.hidden_dim, num_classes=config.num_classes, dropout=config.dropout)
    model.to(device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lr=config.learning_rate, weight_decay=config.weight_decay, params=model.parameters())

    return model, loss_fn, optimizer

    #raise NotImplementedError("Return the model, loss function, and optimizer with weight decay.")


# Run one full training epoch.
# Return one summary for the whole training epoch.
# Do not return one value per sample.
# Use the whole train loader, then return the mean loss and mean accuracy.
# Return a dict with at least:
# - "loss"
# - "accuracy"
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:

    losses, accuracies = [], []

    model.train()

    for x_data, y_targets in loader:
        x_data = x_data.to(device)
        y_targets = y_targets.to(device)

        optimizer.zero_grad()

        logits = model(x_data)
        loss = loss_fn(logits, y_targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        max_indexes = torch.argmax(logits, dim=1) #across columns # or you can just say logits.argmax
        # The below is the same as counting where prediction equals target in the batch and then deviding by batch_size(num_elements)
        # to get the accuracy.
        # When you think about it, the below will have an array of 1's where pred == y_target for that row/sample.
        # Then when you take the mean, it just adds all the 1's and 0's and divides by num_elements.
        # Because we used boolean which became 1's and 0's, when summed up, the 0's wont matter. Adding up the 1's
        # is the same as counting the number of elements where pred=target, and the mean is just dividing by the num_elements/size.
        correct = (max_indexes == y_targets).float()
        mean_accuracy_accross_batch = float(correct.mean())
        accuracies.append(mean_accuracy_accross_batch)

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()

    return {"loss": mean_loss, "accuracy":mean_acc}

    #raise NotImplementedError("Return one epoch of training metrics with loss and accuracy.")


# Run one full validation epoch.
# Return one summary for the whole loader.
# If you pass the validation loader, return the mean validation loss and mean validation accuracy.
# If you pass the test loader, return the mean test loss and mean test accuracy.
# Do not return one value per sample.
# Return a dict with at least:
# - "loss"
# - "accuracy"
@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    model.eval()

    losses, accuracies = [], []

    for x_data, y_targets in loader:

        x_data = x_data.to(device)
        y_targets = y_targets.to(device)

        logits = model(x_data)
        loss = loss_fn(logits, y_targets)

        losses.append(loss.item())
        
        max_indexes = logits.argmax(dim=1) #max column index per row/sample
        correct = (max_indexes == y_targets).float()
        mean_accuracy_across_batch = correct.mean().item()
        accuracies.append(mean_accuracy_across_batch)

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()
    
    return {"loss": mean_loss, "accuracy":mean_acc}
    #raise NotImplementedError("Return one epoch of validation metrics with loss and accuracy.")


# Decide whether to stop.
# Keep the rule simple.
# A common answer is:
# - stop when epochs_without_improvement >= patience
# - and current_val_loss did not beat best_val_loss
def should_stop_early(best_val_loss: float, current_val_loss: float, epochs_without_improvement: int, patience: int) -> bool:

    if epochs_without_improvement >= patience and best_val_loss < current_val_loss:
        return True
    
    return False

    #raise NotImplementedError("Return True when validation loss has not improved for patience epochs.")


# Run the full experiment and keep history.
# Each item in these lists should be one epoch-level mean.
# Example:
# - train_losses[i] is the mean training loss for epoch i
# - val_losses[i] is the mean validation loss for epoch i
# Return a dict with at least these keys:
# - "train_losses"
# - "val_losses"
# You can also include:
# - "train_accuracies"
# - "val_accuracies"
# - "best_val_loss"
# - "stopped_early"
def run_training(config: RegularizationConfig) -> dict[str, list[float] | float | bool]:
    #raise NotImplementedError("Run training, track validation history, and include whether early stopping happened.")

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    stopped_early = False
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = sys.float_info.max
    counter = 0
    for i in range(config.epochs):
        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = validate_one_epoch(model, loaders["val"], loss_fn)

        train_losses.append(train_dict["loss"])
        train_accs.append(train_dict["accuracy"])
        val_losses.append(val_dict["loss"])
        val_accs.append(val_dict["accuracy"])

        current_val_loss = val_dict["loss"]

        if should_stop_early(best_val_loss, current_val_loss, counter, config.patience):
            stopped_early = True
            break

        if current_val_loss < best_val_loss:
            #this means the loss improved
            best_val_loss = current_val_loss
            #reset counter
            counter=0
        else:
            counter+=1

        if i%10 == 0:
            print(f"Epoch {i} -> loss: {train_dict["loss"]}, acc: {train_dict["accuracy"]}, val_loss: {val_dict["loss"]}, val_acc: {val_dict["accuracy"]}")

    print(f"Best Val Loss: {best_val_loss}, early_stopped_at_epoch: {i}")

    return {"train_losses":train_losses, "val_losses":val_losses, 
            "train_accuracies": train_accs, "val_accuracies": val_accs,
            "best_val_loss": best_val_loss, "stopped_early": stopped_early}


def check_metrics(metrics: dict[str, list[float] | float | bool]) -> bool:
    """Return True when the history is ready for plotting and inspection."""
    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    return bool(train_losses) and bool(val_losses)



def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False



def main() -> None:
    """Run the regularization scaffold one step at a time."""
    config = RegularizationConfig()
    print("Running exercise 06 starter scaffold...")
    print("Main goal: notice overfitting and respond with validation-based decisions.")
    print("Hint: use plot_loss_curves(train_losses, val_losses) after your history is ready.")

    completed_steps = 0
    total_steps = 8


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("RegularizedMLP.__init__", RegularizedMLP, config.num_features, config.hidden_dim, config.num_classes, config.dropout)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("RegularizedMLP.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    train_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, train_model, loaders["train"], loss_fn, optimizer)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("validate_one_epoch", validate_one_epoch, train_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("should_stop_early", should_stop_early, 1.0, 1.1, config.patience, config.patience)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "regularization history", float(int(passed)), "non-empty train/val curves")

        train_losses = metrics.get("train_losses", [])
        val_losses = metrics.get("val_losses", [])
        if train_losses and val_losses:
            plot_loss_curves(train_losses, val_losses, title="Basics exercise 06 Loss Curves")


if __name__ == "__main__":
    main()














