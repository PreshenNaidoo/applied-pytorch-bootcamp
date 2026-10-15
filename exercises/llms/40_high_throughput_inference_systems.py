"""
============================================================
EXERCISE 40: HIGH THROUGHPUT INFERENCE SYSTEMS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny simulation of language-model inference serving choices.

Plain-language meanings:
- latency means how long one user waits
- throughput means how many tokens or requests are served per second
- batching means serving several requests together
- streaming means sending tokens as they are generated
- KV cache memory is memory used to store old keys and values during generation
- paged attention stores KV cache in blocks so memory is easier to reuse
- speculative decoding drafts tokens with a small model and checks them with a larger model

WHY THIS EXERCISE EXISTS
Frontier labs do not only train models.
They also need to serve them quickly and cheaply.

This exercise teaches:
- request queues
- batch formation
- latency versus throughput
- KV cache memory estimates
- streaming token events
- speculative decoding acceptance rate

HOW THIS FITS INTO THE BOOTCAMP
The basic inference exercise served simple requests.
This file focuses on language-model generation systems.

Beginner intuition:
- batching can improve throughput but may make one user wait longer
- streaming makes the user see progress before the full answer is done
- KV cache speeds generation but uses memory
- speculative decoding can help if draft tokens are often accepted

DATASET
No real model dataset is required.

Use a tiny local request dataset:
- a list of fake user generation requests
- each request has:
  - request id
  - prompt text
  - prompt token count
  - requested generation length

Why this is enough:
- this exercise is about serving behavior
- request shape and token counts are enough to study batching, latency, and memory

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- write several request dictionaries
- place them in a queue
- group them into batches
- simulate generation time
- simulate streaming token events
- simulate speculative decoding with draft and verified token lists

INPUTS AND OUTPUTS
Input:
- request queue
- model shape settings
- generation length settings

Output:
- number of batches
- tokens per second
- KV cache memory estimate
- streaming events
- speculative decoding acceptance rate

STEP-BY-STEP TODO
1. Build toy generation requests.
2. Form batches from a queue.
3. Simulate token generation time.
4. Measure latency.
5. Measure throughput.
6. Estimate KV cache memory.
7. Simulate streaming output events.
8. Simulate speculative decoding acceptance.
9. Write a serving tradeoff note.

DEFINITION OF DONE
You are done when:
- batch latency and throughput are reported
- KV cache memory is estimated
- streaming events are returned
- speculative decoding has an acceptance rate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again


GenerationRequest = dict[str, str | int]
RequestBatch = list[GenerationRequest]
BatchGenerationRow = dict[str, float | int]
StreamingEvent = dict[str, str | int | float]
SpeculativeDecodingStats = dict[str, float | int]
InferenceSystemSummary = dict[str, int | float | str | list[StreamingEvent]]
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.

# Store settings for the inference systems simulation.
# Expected return:
# - a config object with batching, generation, and KV-cache settings
@dataclass
class HighThroughputInferenceConfig:
    """Settings for LLM inference serving simulation."""

    max_batch_size: int = 4
    prompt_tokens_per_request: int = 64
    generated_tokens_per_request: int = 32
    model_dim: int = 512
    num_layers: int = 8
    num_kv_heads: int = 8
    head_dim: int = 64
    bytes_per_cache_value: int = 2


# Build toy generation requests.
# Expected return:
# - list of request dictionaries
def build_generation_requests() -> list[GenerationRequest]:
    """Return toy generation requests."""
    # Goal:
    # - Return a small queue of fake user requests for text generation.
    # - Each request should say how many prompt tokens already exist and how many new tokens the user wants.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Return `list[GenerationRequest]`.
    # - Each request must have exactly these keys:
    #   - `"request_id"`: int, such as `0`
    #   - `"prompt"`: str, such as `"Explain batching in one sentence."`
    #   - `"prompt_tokens"`: int, number of tokens already in the prompt
    #   - `"new_tokens"`: int, number of tokens to generate
    # - Small example:
    #   `{"request_id": 0, "prompt": "Define latency.", "prompt_tokens": 8, "new_tokens": 12}`
    #
    # Method:
    # - Use a plain Python list of dictionaries. No model or tokenizer is needed.
    # - Use integer token counts directly because this is a serving simulation.
    #
    # Steps:
    # 1. Write at least 6 requests so batching has more than one batch.
    # 2. Give request ids in order, starting from 0.
    # 3. Use positive integers for `prompt_tokens` and `new_tokens`.
    # 4. Keep prompts short so the data is easy to inspect.
    #
    # Checks:
    # - `len(requests) >= 6`
    # - `set(requests[0]) == {"request_id", "prompt", "prompt_tokens", "new_tokens"}`
    # - `requests[0]["prompt_tokens"] > 0`
    # - `requests[0]["new_tokens"] > 0`
    #
    raise NotImplementedError("Return requests with ids, prompts, and generation lengths.")


# Form batches from queued requests.
# Expected return:
# - list of batches, where each batch is a list of requests
def form_request_batches(
    requests: list[GenerationRequest],
    config: HighThroughputInferenceConfig,
) -> list[RequestBatch]:
    """Group requests into batches."""
    # Goal:
    # - Split the request queue into batches.
    # - A batch is a list of requests served together.
    #
    # Input:
    # - `requests`: list from `build_generation_requests`.
    # - `config.max_batch_size`: largest number of requests allowed in one batch.
    # - Example with `max_batch_size=2`:
    #   requests `[r0, r1, r2, r3, r4]` become `[[r0, r1], [r2, r3], [r4]]`.
    #
    # Output:
    # - Return `list[RequestBatch]`.
    # - Each batch length must be at most `config.max_batch_size`.
    # - Keep the original request order.
    #
    # Method:
    # - Use Python slicing:
    #   `requests[start : start + config.max_batch_size]`
    # - Use `range(0, len(requests), config.max_batch_size)` for the start indexes.
    #
    # Steps:
    # 1. Start with an empty `batches` list.
    # 2. Loop over start indexes in steps of `config.max_batch_size`.
    # 3. Slice the requests for each batch.
    # 4. Append each non-empty batch.
    # 5. Return all batches.
    #
    # Checks:
    # - `all(len(batch) <= config.max_batch_size for batch in batches)`
    # - `sum(len(batch) for batch in batches) == len(requests)`
    # - `batches[0][0]["request_id"] == requests[0]["request_id"]`
    #
    raise NotImplementedError("Create batches up to max_batch_size.")


# Estimate KV cache memory.
# Expected return:
# - memory in megabytes
def estimate_generation_kv_cache_mb(
    batch_size: int,
    total_tokens: int,
    config: HighThroughputInferenceConfig,
) -> float:
    """Estimate KV cache memory for generation."""
    # Goal:
    # - Estimate how much memory the KV cache uses for a generation batch.
    # - KV cache stores keys and values for past tokens so the model does not recompute them.
    #
    # Input:
    # - `batch_size`: number of requests in the batch.
    # - `total_tokens`: prompt tokens plus generated tokens per request.
    # - `config.num_layers`: number of transformer layers.
    # - `config.num_kv_heads`: number of key/value heads per layer.
    # - `config.head_dim`: values stored per key or value head.
    # - `config.bytes_per_cache_value`: bytes for one stored number, such as 2 for float16.
    #
    # Output:
    # - Return one float: estimated KV cache memory in megabytes.
    # - Use MB as `bytes / (1024 * 1024)`.
    #
    # Method:
    # - Formula:
    #   `num_values = batch_size * total_tokens * config.num_layers * 2 * config.num_kv_heads * config.head_dim`
    #   `bytes_total = num_values * config.bytes_per_cache_value`
    #   `mb = bytes_total / (1024 * 1024)`
    # - The `2` means each token stores both K and V. K means key, V means value.
    #
    # Steps:
    # 1. Compute `num_values` with the formula above.
    # 2. Convert values to bytes with `config.bytes_per_cache_value`.
    # 3. Convert bytes to MB.
    # 4. Return the MB value as a float.
    #
    # Checks:
    # - Tiny hand check: if all counts are 1, except the K/V factor is 2 and bytes are 2,
    #   then `bytes_total` is 4 bytes.
    # - For real config values, the returned MB should be positive when batch and token counts are positive.
    #
    raise NotImplementedError("Return KV cache memory in MB.")


# Simulate one batch generation.
# Expected return:
# - dictionary with latency and generated token count
def simulate_batch_generation(
    batch: RequestBatch,
    config: HighThroughputInferenceConfig,
) -> BatchGenerationRow:
    """Simulate generation for one batch."""
    # Goal:
    # - Simulate how long one batch takes and how many new tokens it creates.
    # - This is a toy timing model, not a real benchmark.
    #
    # Input:
    # - `batch`: one batch from `form_request_batches`.
    # - Each request has `"prompt_tokens"` and `"new_tokens"`.
    # - `config.prompt_tokens_per_request` and `config.generated_tokens_per_request`
    #   are fallback values if you choose to use the config for simple timing.
    #
    # Output:
    # - Return `BatchGenerationRow`.
    # - Exact keys:
    #   - `"batch_size"`: int, number of requests in the batch
    #   - `"generated_tokens"`: int, sum of requested new tokens in the batch
    #   - `"latency_seconds"`: float, simulated batch time
    #   - `"kv_cache_mb"`: float, estimated cache memory for this batch
    #
    # Method:
    # - Use simple formulas:
    #   `batch_size = len(batch)`
    #   `generated_tokens = sum(request["new_tokens"] for request in batch)`
    #   `max_total_tokens = max(request["prompt_tokens"] + request["new_tokens"] for request in batch)`
    #   `latency_seconds = 0.02 + 0.004 * max_total_tokens`
    # - The fixed `0.02` is startup overhead. The `0.004` is fake seconds per token.
    # - Call:
    #   `estimate_generation_kv_cache_mb(batch_size, max_total_tokens, config)`
    #
    # Steps:
    # 1. Count the requests in the batch.
    # 2. Sum all requested new tokens.
    # 3. Find the largest prompt-plus-generation length in the batch.
    # 4. Estimate latency with the formula above.
    # 5. Estimate KV cache memory.
    # 6. Return the four exact keys listed above.
    #
    # Checks:
    # - `row["batch_size"] == len(batch)`
    # - `row["generated_tokens"] > 0`
    # - `row["latency_seconds"] > 0.0`
    # - `row["kv_cache_mb"] > 0.0`
    #
    raise NotImplementedError("Return batch latency and token count.")


# Compute throughput from simulation rows.
# Expected return:
# - tokens per second
def compute_tokens_per_second(batch_rows: list[BatchGenerationRow]) -> float:
    """Compute generated tokens per second."""
    # Goal:
    # - Compute how many generated tokens the server produces per second.
    #
    # Input:
    # - `batch_rows`: rows from `simulate_batch_generation`.
    # - Each row has:
    #   `{"generated_tokens": int, "latency_seconds": float, ...}`
    #
    # Output:
    # - Return one float: generated tokens per second.
    #
    # Method:
    # - Formula:
    #   `total_tokens = sum(row["generated_tokens"] for row in batch_rows)`
    #   `total_seconds = sum(row["latency_seconds"] for row in batch_rows)`
    #   `tokens_per_second = total_tokens / total_seconds`
    #
    # Steps:
    # 1. If `batch_rows` is empty, return `0.0`.
    # 2. Sum generated tokens across rows.
    # 3. Sum latency seconds across rows.
    # 4. If `total_seconds` is 0, return `0.0` to avoid division by zero.
    # 5. Return `total_tokens / total_seconds`.
    #
    # Checks:
    # - With rows `[{"generated_tokens": 10, "latency_seconds": 2.0}]`,
    #   the answer should be `5.0`.
    # - The returned value should never be negative.
    #
    raise NotImplementedError("Compute throughput from generated tokens and time.")


# Simulate streaming token events.
# Expected return:
# - list of event dictionaries
def simulate_streaming_events(request: GenerationRequest) -> list[StreamingEvent]:
    """Return token streaming events for one request."""
    # Goal:
    # - Return one streaming event for each generated token in one request.
    # - Streaming means the user sees tokens as they arrive, not only at the end.
    #
    # Input:
    # - `request`: one `GenerationRequest`.
    # - Example:
    #   `{"request_id": 0, "prompt": "Define latency.", "prompt_tokens": 8, "new_tokens": 3}`
    #
    # Output:
    # - Return `list[StreamingEvent]`.
    # - There should be exactly `request["new_tokens"]` events.
    # - Each event must have exactly these keys:
    #   - `"request_id"`: copied from the request
    #   - `"token_index"`: int starting at 0
    #   - `"token_text"`: fake token text, such as `"tok_0"`
    #   - `"time_seconds"`: fake time when this token is sent
    #
    # Method:
    # - Use a loop:
    #   `for token_index in range(request["new_tokens"]):`
    # - Use a simple timing formula:
    #   `time_seconds = 0.03 + 0.01 * (token_index + 1)`
    # - The `0.03` is fake first-token delay. The `0.01` is fake time between tokens.
    #
    # Steps:
    # 1. Start with an empty `events` list.
    # 2. Loop from token index 0 to `new_tokens - 1`.
    # 3. Build one event dictionary per token.
    # 4. Append each event.
    # 5. Return the events list.
    #
    # Checks:
    # - `len(events) == request["new_tokens"]`
    # - `events[0]["token_index"] == 0`
    # - Event times should increase as token indexes increase.
    #
    raise NotImplementedError("Return one event per generated token.")


# Simulate speculative decoding.
# Expected return:
# - dictionary with drafted tokens, accepted tokens, and acceptance rate
def simulate_speculative_decoding(draft_tokens: list[str], verified_tokens: list[str]) -> SpeculativeDecodingStats:
    """Return speculative decoding acceptance stats."""
    # Goal:
    # - Compare tokens proposed by a small draft model with tokens accepted by the main model.
    # - Speculative decoding is useful when many draft tokens match the main model.
    #
    # Input:
    # - `draft_tokens`: tokens proposed by the small draft model.
    #   Example: `["The", "cat", "sat"]`
    # - `verified_tokens`: tokens chosen by the main model.
    #   Example: `["The", "cat", "ran"]`
    #
    # Output:
    # - Return `SpeculativeDecodingStats`.
    # - Exact keys:
    #   - `"drafted_tokens"`: number of draft tokens checked
    #   - `"accepted_tokens"`: number of matching tokens before the first mismatch
    #   - `"acceptance_rate"`: `accepted_tokens / drafted_tokens`
    #
    # Method:
    # - Compare tokens in order with:
    #   `for draft_token, verified_token in zip(draft_tokens, verified_tokens):`
    # - Stop counting at the first mismatch.
    # - Formula:
    #   `acceptance_rate = accepted_tokens / drafted_tokens`
    #
    # Steps:
    # 1. Set `drafted_tokens = len(draft_tokens)`.
    # 2. If there are no draft tokens, return zeros for all fields.
    # 3. Start `accepted_tokens` at 0.
    # 4. Compare each draft token with the verified token at the same position.
    # 5. Add 1 for each match.
    # 6. Stop when the first mismatch appears.
    # 7. Compute acceptance rate.
    #
    # Checks:
    # - With draft `["The", "cat", "sat"]` and verified `["The", "cat", "ran"]`,
    #   accepted tokens should be `2` and acceptance rate should be `2 / 3`.
    # - `0.0 <= stats["acceptance_rate"] <= 1.0`
    #
    raise NotImplementedError("Compare draft tokens with verified tokens.")


# Run the full inference systems workflow.
# Expected return:
# - summary dictionary with latency, throughput, memory, streaming, and speculation stats
def run_high_throughput_inference_demo(config: HighThroughputInferenceConfig) -> InferenceSystemSummary:
    """Run the high-throughput inference simulation."""
    # Goal:
    # - Run the whole serving simulation and return one summary.
    # - This function connects request data, batching, timing, KV memory, streaming, and speculative decoding.
    #
    # Input:
    # - `config`: `HighThroughputInferenceConfig` with batch size and KV-cache settings.
    #
    # Output:
    # - Return `InferenceSystemSummary`.
    # - Exact keys:
    #   - `"batch_count"`: int
    #   - `"tokens_per_second"`: float
    #   - `"kv_cache_mb"`: float, total or peak cache memory for a batch
    #   - `"streaming_events"`: list of streaming event dictionaries
    #   - `"speculative_acceptance_rate"`: float
    #   - `"serving_tradeoff_note"`: short plain-English note
    #
    # Method:
    # - Call the helpers in this order:
    #   `build_generation_requests()`
    #   `form_request_batches(requests, config)`
    #   `simulate_batch_generation(batch, config)` for each batch
    #   `compute_tokens_per_second(batch_rows)`
    #   `simulate_streaming_events(requests[0])`
    #   `simulate_speculative_decoding(draft_tokens, verified_tokens)`
    #
    # Steps:
    # 1. Build the toy requests.
    # 2. Batch the requests.
    # 3. Simulate each batch and collect `batch_rows`.
    # 4. Compute throughput from `batch_rows`.
    # 5. Choose a KV cache number. A useful simple choice is:
    #    `kv_cache_mb = max(row["kv_cache_mb"] for row in batch_rows)`
    # 6. Simulate streaming for the first request.
    # 7. Use small token lists for speculative decoding, such as:
    #    `draft_tokens = ["The", "answer", "is", "short"]`
    #    `verified_tokens = ["The", "answer", "is", "clear"]`
    # 8. Return the six summary keys listed above.
    #
    # Checks:
    # - `summary["batch_count"] == len(batches)`
    # - `summary["tokens_per_second"] >= 0.0`
    # - `len(summary["streaming_events"]) == requests[0]["new_tokens"]`
    # - `0.0 <= summary["speculative_acceptance_rate"] <= 1.0`
    #
    raise NotImplementedError("Run batching, memory, streaming, and speculative decoding checks.")


# Check the summary.
# Expected return:
# - True when key outputs are present
def check_metrics(metrics: InferenceSystemSummary) -> bool:
    """Return True when inference-system outputs are present."""
    required = {"batch_count", "tokens_per_second", "kv_cache_mb", "streaming_events", "speculative_acceptance_rate", "serving_tradeoff_note"}
    return required.issubset(metrics)


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the high-throughput inference scaffold."""
    config = HighThroughputInferenceConfig()

    try:
        metrics = run_high_throughput_inference_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    print(metrics)
    print_pass_or_try_again(check_metrics(metrics), "inference_system_outputs", int(check_metrics(metrics)), "1")


if __name__ == "__main__":
    main()
