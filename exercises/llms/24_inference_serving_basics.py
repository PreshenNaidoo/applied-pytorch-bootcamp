"""
============================================================
EXERCISE 24: INFERENCE SERVING BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny inference service shape.

Plain-language meaning:
- inference means using a trained model to make predictions
- serving means making that prediction code usable by another program
- latency means how long one request takes
- throughput means how many requests you handle per second

WHY THIS EXERCISE EXISTS
Research code is more useful when other people can run it.
Even a simple model should have a clean prediction path.
This exercise teaches:
- loading a saved model
- writing a `predict_one` function
- batching requests
- timing latency
- simple request and response dictionaries
- safe error handling

HOW THIS FITS INTO THE BOOTCAMP
Many earlier files focused on training.
This file asks the next practical question:
- how do you make a trained model usable by other code?

What each step is doing in simple words:
1. load a model in inference mode
2. check that one request is valid
3. turn the request into model input
4. run one prediction
5. turn model output into a clean response
6. repeat for batches
7. time the result so the serving path has real performance numbers

Beginner intuition:
- training code teaches the model
- serving code lets other code use the model
- a useful serving path should be clear, predictable, and easy to debug

Small example:
- one request comes in as a Python dictionary
- preprocessing turns it into a tensor
- the model predicts
- postprocessing turns that prediction back into a clean response dictionary

DATASET
Use a tiny real classification setup.

Clear default:
- `sklearn.datasets.load_wine`
- request input: 13 numeric wine features
- response output: predicted class id and class probabilities
- use a small MLP trained on the wine data

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- train or load one small model on `sklearn.datasets.load_wine`
- define the request format clearly
- define the response format clearly
- test one request and one batch

STEP-BY-STEP TODO
1. Build or load a small trained model.
2. Write preprocessing for one request.
3. Write `predict_one(...)`.
4. Write `predict_batch(...)`.
5. Return clean response dictionaries.
6. Measure latency for one request.
7. Measure throughput for a batch.
8. Add simple input checks.
9. Write a short serving note.

DEFINITION OF DONE
You are done when:
- one request can be predicted
- a batch can be predicted
- latency is reported
- throughput is reported
- bad inputs return a clear error

INPUTS AND OUTPUTS
Input:
- request dictionary
- or list of request dictionaries
Output:
- response dictionary
- latency number
- throughput summary

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use one small trained classifier.
The point is the serving path, not model training.

LOSS FUNCTION
No training loss is required for the serving exercise itself.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class InferenceServingConfig:
    """Settings for inference serving basics."""
    input_dim: int = 13
    hidden_dim: int = 64
    num_classes: int = 3
    batch_size: int = 16
    random_state: int = 42
    checkpoint_path: str = str(BOOTCAMP_ROOT / "outputs" / "serving_model.pt")

def describe_serving_plan(config: InferenceServingConfig) -> dict[str, object]:
    """Return the serving pieces this exercise should build."""
    return {
        "request_shape": {"features": config.input_dim},
        "response_keys": ["predicted_class", "confidence", "latency_ms"],
        "batch_size": config.batch_size,
    }


# Beginner hints for this file:
# - training code and serving code are not the same job
# - serving should focus on clean inputs, clean outputs, and predictable behavior
# - latency means how long one request takes from start to finish
# - throughput means how many requests can be handled in a second
# - always validate inputs early so failures are easier to understand
class ServingMLP(nn.Module):
    """Small model used for inference serving practice."""
    def __init__(self, config: InferenceServingConfig) -> None:
        # Goal:
        # Build a small classifier for serving.
        #
        # Input:
        # - `config`: InferenceServingConfig
        #
        # Output:
        # - no return value
        # - creates `self.net`
        #
        # Shape plan:
        # - input batch: `[batch_size, 13]`
        # - hidden batch: `[batch_size, config.hidden_dim]`
        # - logits: `[batch_size, 3]`
        #
        # Method:
        # Use:
        # ```python
        # self.net = nn.Sequential(
        #     nn.Linear(config.input_dim, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.num_classes),
        # )
        # ```
        #
        # Meaning:
        # - Wine has 13 numeric input features.
        # - Wine has 3 classes.
        # - The model returns raw logits. Softmax is done after inference.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.net`.
        # 3. Leave `.to(device)` to `load_model_for_inference`.
        #
        # Checks:
        # - Input shape `[8, 13]` should return logits shaped `[8, 3]`.
        # - Do not apply softmax inside the model.
        super().__init__()
        raise NotImplementedError("Build a small classifier for serving.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        # Goal:
        # Return raw logits for each wine class.
        #
        # Input:
        # - `x`: float tensor shaped `[batch_size, config.input_dim]`
        #
        # Output:
        # - tensor shaped `[batch_size, config.num_classes]`
        #
        # Method:
        # Use:
        # `return self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through `self.net`.
        # 2. Return the logits.
        #
        # Checks:
        # - Return logits, not probabilities.
        # - Batch size should stay the same.
        raise NotImplementedError("Return raw logits.")

# Load or create a model ready for inference.
# The key serving check is that the model should be in eval mode before prediction.
def load_model_for_inference(config: InferenceServingConfig) -> nn.Module:
    """Load or create a model ready for inference."""
    # Goal:
    # Load model weights and set the model to eval mode.
    # If no checkpoint exists yet, create the model and leave it untrained.
    #
    # Input:
    # - `config`: InferenceServingConfig
    #
    # Output:
    # - `nn.Module`
    # - model on `device`
    # - model in eval mode
    #
    # Method:
    # Use:
    # ```python
    # model = ServingMLP(config).to(device)
    # checkpoint_path = Path(config.checkpoint_path)
    # if checkpoint_path.exists():
    #     state_dict = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(state_dict)
    # model.eval()
    # return model
    # ```
    #
    # Meaning:
    # - `map_location=device` loads weights onto CPU or GPU correctly.
    # - `model.eval()` turns off training-only behavior such as dropout.
    # - This model does not include dropout, but eval mode is still the right serving habit.
    #
    # Steps:
    # 1. Build `ServingMLP(config)`.
    # 2. Move it to `device`.
    # 3. If `config.checkpoint_path` exists, load the state dict.
    # 4. Call `model.eval()`.
    # 5. Return the model.
    #
    # Checks:
    # - `not model.training` should be true.
    # - `next(model.parameters()).device.type` should match `device`.
    # - Wine data has 13 input features and 3 classes, matching the config defaults.
    raise NotImplementedError("Load model weights and set the model to eval mode.")

# Validate one request before it reaches the model.
# Good serving code fails early and clearly on bad inputs.
def validate_request(request: dict[str, object], config: InferenceServingConfig) -> dict[str, object]:
    """Check one request and return a clean result or error."""
    # Goal:
    # Validate that the request has numeric features of the right length.
    # Return a clean result instead of letting a tensor error happen later.
    #
    # Input:
    # - `request`: dict[str, object]
    # - `config`: InferenceServingConfig
    #
    # Output:
    # Return one of these dictionaries:
    # - valid request: `{"ok": True, "features": features}`
    # - invalid request: `{"ok": False, "error": "short plain message"}`
    #
    # Request format:
    # ```python
    # {"features": [float_1, float_2, ..., float_13]}
    # ```
    #
    # Small valid example:
    # ```python
    # {"features": [0.0] * config.input_dim}
    # ```
    #
    # Method:
    # Check:
    # - key `"features"` exists
    # - the value is a `list` or `tuple`
    # - length is `config.input_dim`
    # - every value can be converted to `float`
    #
    # Steps:
    # 1. Check that `"features"` is present.
    # 2. Check that it is a list or tuple.
    # 3. Check the length.
    # 4. Try `features = [float(value) for value in raw_features]`.
    # 5. Return the valid or invalid dictionary.
    #
    # Checks:
    # - Bad inputs should not raise confusing PyTorch errors.
    # - A valid result should include numeric features of length `config.input_dim`.
    raise NotImplementedError("Validate that the request has numeric features of the right length.")

# Turn one request into a model input tensor.
# Return shape:
# - `[1, input_dim]`
def preprocess_request(request: dict[str, object], config: InferenceServingConfig) -> torch.Tensor:
    """Turn one request into a model input tensor."""
    # Goal:
    # Convert one valid request into a tensor shaped `[1, input_dim]`.
    #
    # Input:
    # - `request`: dict[str, object]
    # - `config`: InferenceServingConfig
    #
    # Output:
    # - `torch.Tensor`
    # - shape `[1, config.input_dim]`
    # - dtype `torch.float32`
    # - moved to `device`
    #
    # Method:
    # Use:
    # ```python
    # validation = validate_request(request, config)
    # if not validation["ok"]:
    #     raise ValueError(str(validation["error"]))
    # features = validation["features"]
    # tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    # ```
    #
    # Meaning:
    # - `torch.tensor(features, dtype=torch.float32)` creates shape `[13]`.
    # - `.unsqueeze(0)` adds a batch dimension, making shape `[1, 13]`.
    #
    # Steps:
    # 1. Validate the request.
    # 2. Raise `ValueError` if it is invalid.
    # 3. Convert features to a float tensor.
    # 4. Add the batch dimension.
    # 5. Move the tensor to `device`.
    # 6. Return it.
    #
    # Checks:
    # - A valid request should return shape `(1, config.input_dim)`.
    # - The dtype should be `torch.float32`.
    raise NotImplementedError("Convert one request into a tensor with shape [1, input_dim].")

def postprocess_logits(logits: torch.Tensor) -> dict[str, object]:
    """Turn model logits into a response dictionary."""
    # Goal:
    # Return predicted class, confidence, and all class probabilities.
    #
    # Input:
    # - `logits`: tensor shaped `[1, config.num_classes]` or `[config.num_classes]`
    #
    # Output:
    # Return this exact dictionary:
    # - `"predicted_class"`: int
    # - `"confidence"`: float
    # - `"class_probabilities"`: list[float]
    #
    # Method:
    # Use:
    # ```python
    # if logits.ndim == 2:
    #     logits = logits.squeeze(0)
    # probabilities = torch.softmax(logits, dim=-1)
    # predicted_class = int(probabilities.argmax(dim=-1).item())
    # confidence = float(probabilities[predicted_class].item())
    # ```
    #
    # Meaning:
    # - Softmax turns raw logits into probabilities that sum to `1.0`.
    # - `argmax` finds the class with the highest probability.
    #
    # Steps:
    # 1. Remove the batch dimension if the shape is `[1, num_classes]`.
    # 2. Apply softmax on the class dimension.
    # 3. Pick the predicted class with argmax.
    # 4. Read the predicted class probability as confidence.
    # 5. Convert probabilities to a Python list.
    # 6. Return the response dictionary.
    #
    # Checks:
    # - Probabilities should sum to about `1.0`.
    # - `confidence` should be between `0.0` and `1.0`.
    raise NotImplementedError("Return predicted class and confidence.")

# Run one request end to end.
# Return a clean response dictionary that includes latency.
def predict_one(model: nn.Module, request: dict[str, object], config: InferenceServingConfig) -> dict[str, object]:
    """Run one prediction request."""
    # Goal:
    # Return one clean prediction response with latency.
    # Bad inputs should return a clean error dictionary.
    #
    # Input:
    # - `model`: nn.Module
    # - `request`: dict[str, object]
    # - `config`: InferenceServingConfig
    #
    # Output:
    # Valid response:
    # - `"ok": True`
    # - `"predicted_class"`: int
    # - `"confidence"`: float
    # - `"class_probabilities"`: list[float]
    # - `"latency_ms"`: float
    #
    # Invalid response:
    # - `"ok": False`
    # - `"error"`: short plain message
    # - `"latency_ms"`: float
    #
    # Method:
    # Use:
    # ```python
    # start = time.perf_counter()
    # validation = validate_request(request, config)
    # if not validation["ok"]:
    #     latency_ms = (time.perf_counter() - start) * 1000.0
    #     return {"ok": False, "error": validation["error"], "latency_ms": latency_ms}
    # inputs = preprocess_request(request, config)
    # model.eval()
    # with torch.no_grad():
    #     logits = model(inputs)
    # response = postprocess_logits(logits)
    # response["ok"] = True
    # response["latency_ms"] = (time.perf_counter() - start) * 1000.0
    # ```
    #
    # Meaning:
    # - `torch.no_grad()` saves memory because serving does not need gradients.
    # - Latency is measured from the start of request handling to the final response.
    #
    # Steps:
    # 1. Start the timer.
    # 2. Validate the request.
    # 3. Return a clean error if validation fails.
    # 4. Preprocess the request.
    # 5. Run the model under `torch.no_grad()`.
    # 6. Postprocess logits.
    # 7. Add `ok` and latency fields.
    # 8. Return the response.
    #
    # Checks:
    # - Valid responses should include `"ok": True`.
    # - Invalid responses should not call the model.
    # - `latency_ms` should be non-negative.
    _ = time.perf_counter
    raise NotImplementedError("Return one clean prediction response with latency.")

def predict_batch(model: nn.Module, requests: list[dict[str, object]], config: InferenceServingConfig) -> list[dict[str, object]]:
    """Run a batch of prediction requests."""
    # Goal:
    # Return one response per request.
    # Keep the output order the same as the input order.
    #
    # Input:
    # - `model`: nn.Module
    # - `requests`: list[dict[str, object]]
    # - `config`: InferenceServingConfig
    #
    # Output:
    # - `list[dict[str, object]]`
    # - length equals `len(requests)`
    # - each item is the same response shape returned by `predict_one`
    #
    # Method:
    # Simple path:
    # ```python
    # return [predict_one(model, request, config) for request in requests]
    # ```
    #
    # Optional faster path:
    # - Validate all requests first.
    # - Stack inputs into shape `[batch_size, input_dim]`.
    # - Run the model once.
    # - Postprocess each row of logits.
    #
    # For this exercise, the simple path is enough because it is clearer.
    #
    # Steps:
    # 1. Create an empty response list.
    # 2. Loop over requests.
    # 3. Call `predict_one` for each request.
    # 4. Append each response.
    # 5. Return the list.
    #
    # Checks:
    # - `len(responses)` should equal `len(requests)`.
    # - Bad requests should produce error responses, not crash the batch.
    raise NotImplementedError("Return one response per request.")

# Measure:
# - latency for one request
# - throughput for many requests
# These are the two simplest serving performance numbers.
def benchmark_inference(model: nn.Module, config: InferenceServingConfig) -> dict[str, float]:
    """Measure simple latency and throughput."""
    # Goal:
    # Return latency in milliseconds and throughput in requests per second.
    #
    # Input:
    # - `model`: nn.Module
    # - `config`: InferenceServingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"latency_ms"`: float
    # - `"throughput_requests_per_second"`: float
    #
    # Test request:
    # ```python
    # request = {"features": [0.0] * config.input_dim}
    # ```
    #
    # Method:
    # Measure one-request latency:
    # ```python
    # single_response = predict_one(model, request, config)
    # latency_ms = float(single_response["latency_ms"])
    # ```
    #
    # Measure batch throughput:
    # ```python
    # requests = [request for _ in range(config.batch_size)]
    # start = time.perf_counter()
    # responses = predict_batch(model, requests, config)
    # seconds = time.perf_counter() - start
    # throughput = len(responses) / seconds
    # ```
    #
    # Formula:
    # ```text
    # throughput_requests_per_second = number_of_requests / total_seconds
    # ```
    #
    # Steps:
    # 1. Build one valid request.
    # 2. Measure one call to `predict_one`.
    # 3. Build `config.batch_size` copies of the request.
    # 4. Time `predict_batch`.
    # 5. Compute throughput.
    # 6. Return the dictionary.
    #
    # Checks:
    # Latency and throughput are related but not the same:
    # - latency cares about one request
    # - throughput cares about many requests over time
    # - both values should be positive floats for a valid model
    raise NotImplementedError("Return latency_ms and throughput_requests_per_second.")

def write_serving_note(metrics: dict[str, float]) -> str:
    """Write a short note about serving tradeoffs."""
    # Goal:
    # Explain latency and throughput in plain words.
    #
    # Input:
    # - `metrics`: dict[str, float]
    # - expected keys:
    #   - `"latency_ms"`
    #   - `"throughput_requests_per_second"`
    #
    # Output:
    # - short plain-English string
    #
    # Method:
    # Read:
    # ```python
    # latency_ms = metrics["latency_ms"]
    # throughput = metrics["throughput_requests_per_second"]
    # ```
    #
    # Good note shape:
    # - mention one-request latency
    # - mention batch throughput
    # - say batching can improve throughput but one request still has its own latency
    #
    # Steps:
    # 1. Read latency and throughput from metrics.
    # 2. Build one short note.
    # 3. Return it.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Include the units: milliseconds and requests per second.
    raise NotImplementedError("Explain latency and throughput in plain words.")

def run_serving_demo(config: InferenceServingConfig) -> dict[str, object]:
    """Run the full serving workflow."""
    # Goal:
    # Run `predict_one`, `predict_batch`, benchmark inference, and return a summary.
    #
    # Input:
    # - `config`: InferenceServingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"single_response"`: response dictionary from `predict_one`
    # - `"batch_response_count"`: int
    # - `"latency_ms"`: float
    # - `"throughput"`: float
    # - `"serving_note"`: str
    #
    # Method:
    # Use:
    # ```python
    # set_seed(config.random_state)
    # model = load_model_for_inference(config)
    # request = {"features": [0.0] * config.input_dim}
    # single_response = predict_one(model, request, config)
    # batch_requests = [request for _ in range(config.batch_size)]
    # batch_responses = predict_batch(model, batch_requests, config)
    # metrics = benchmark_inference(model, config)
    # serving_note = write_serving_note(metrics)
    # ```
    #
    # Steps:
    # 1. Set the seed.
    # 2. Load the model for inference.
    # 3. Build one valid request.
    # 4. Run one prediction.
    # 5. Run a batch prediction.
    # 6. Benchmark latency and throughput.
    # 7. Write the serving note.
    # 8. Return the summary dictionary.
    #
    # Checks:
    # - The summary keys should match `check_metrics`.
    # - `batch_response_count` should equal `config.batch_size`.
    # - Bad input handling should be tested separately with `validate_request`.
    raise NotImplementedError("Run predict_one, predict_batch, benchmark, and return a summary.")

def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when serving outputs are present."""
    required = {"single_response", "batch_response_count", "latency_ms", "throughput", "serving_note"}
    return required.issubset(metrics)

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
    """Run the inference serving scaffold."""
    config = InferenceServingConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 4
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_serving_plan", describe_serving_plan, config)
    if not count_or_stop(ok):
        return
    model, ok = report_task_progress("load_model_for_inference", load_model_for_inference, config)
    if not count_or_stop(ok):
        return
    request = {"features": [0.0] * config.input_dim}
    _, ok = report_task_progress("predict_one", predict_one, model, request, config)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_serving_demo", run_serving_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "serving summary", float(int(passed)), "serving keys present")
if __name__ == "__main__":
    main()
