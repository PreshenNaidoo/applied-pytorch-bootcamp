"""
============================================================
EXERCISE 19: RETRIEVAL WITH EMBEDDINGS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny retrieval system.

Plain-language meanings:
- an embedding is a list of numbers that represents an item
- retrieval means finding the closest matching items
- top-k means return the best k matches
- similarity means how close two embeddings are

WHY THIS EXERCISE EXISTS
Retrieval is used in search, RAG, memory systems, data filtering, and multimodal systems.
RAG means retrieval-augmented generation: find useful documents, then give them to a language model.
This exercise teaches:
- how to encode documents
- how to encode queries
- how to build a similarity matrix
- how to return top-k results
- how to measure retrieval quality
- how to inspect failed queries

HOW THIS FITS INTO THE BOOTCAMP
This file extends the earlier text and embedding work.
Instead of predicting one class label,
you now use embeddings to search for the closest useful item.

What each step is doing in simple words:
1. create a tiny searchable document set
2. write queries with one intended match
3. map documents and queries into the same numeric space
4. compare every query with every document
5. return the closest matches
6. check whether the correct document appears near the top
7. read failures so the retrieval numbers mean something real

Beginner intuition:
- retrieval is like search
- you ask a question
- the system finds the closest useful documents
- embeddings make that search numeric

Small example:
- query: `How do I boil pasta?`
- good document: cooking instructions
- bad document: stock market report
- the goal is for the query embedding to land closer to the cooking document than the market document

DATASET
Use the local dummy query/document dataset:
- `data/dual_encoder_retriever_pairs.json`

The JSON file has:
- `pair_count`: the number of query/document pairs
- `pairs`: a list of dictionaries

Each item in `pairs` looks like:
- `{"query": "what is dataloader", "document": "DataLoader groups Dataset items into batches for training."}`

Simple path:
- load the JSON file
- turn each unique document into one retrieval document
- use the matching query as a retrieval query
- store the matching document id as `correct_doc_id`

INPUTS AND OUTPUTS
Document embeddings:
- shape: `[num_documents, embedding_dim]`
Query embeddings:
- shape: `[num_queries, embedding_dim]`
Similarity matrix:
- shape: `[num_queries, num_documents]`
Retrieval output:
- top-k document ids for each query

MODEL ARCHITECTURE (HIGH LEVEL ONLY)

Default path:
- one document encoder
- one query encoder
- similarity function
- top-k retrieval step

Simple mental picture:
- turn documents into points in space
- turn queries into points in the same space
- retrieve the documents that land closest to the query point

METRICS TO TRACK
Track at least:
- top-1 accuracy
- top-5 accuracy
- failed queries
- one short failure note

LOSS FUNCTION
This exercise can be done without training.
If you do train an encoder later, use a contrastive or matching loss.

TARGET RESULT RANGE
Focus on whether the right document appears near the top.

Useful signs:
- top-1 accuracy is meaningfully above random
- top-5 is at least as good as top-1
- failed queries reveal a real similarity problem

COMMON MISTAKES
- comparing unnormalized embeddings when you meant cosine similarity
- mixing up query rows and document columns
- checking whether any document was returned instead of the correct document
- reporting only top-5 when top-1 is very weak

STEP-BY-STEP TODO
1. Load the local JSON pair dataset.
2. Build documents and matching queries from the pairs.
3. Encode documents into embeddings.
4. Encode queries into embeddings.
5. Normalize embeddings if using cosine similarity.
6. Compute a similarity matrix.
7. Return top-k document ids for each query.
8. Measure top-1 and top-5 accuracy.
9. Inspect failed queries.
10. Write a short note about one failure.

DEFINITION OF DONE
You are done when:
- document and query embeddings have the expected shapes
- similarity matrix has shape `[num_queries, num_documents]`
- top-k retrieval works
- top-1 and top-5 accuracy are reported
- you can explain what retrieval is used for
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import string
import sys
from typing import TypedDict
import torch

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

@dataclass
class RetrievalConfig:
    """Settings for the retrieval exercise."""
    embedding_dim: int = 64
    top_k: int = 5
    random_state: int = 42
    use_cosine_similarity: bool = True


class RetrievalPlanSummary(TypedDict):
    """Main shapes and metrics for the retrieval exercise."""

    document_embeddings: str
    query_embeddings: str
    similarity_matrix: str
    top_k: int
    metrics: list[str]


class RetrievalDocument(TypedDict):
    """One searchable document."""

    doc_id: int
    title: str
    text: str


class RetrievalQuery(TypedDict):
    """One query with its correct document id."""

    query: str
    correct_doc_id: int


class RetrievalMetricSummary(TypedDict):
    """Top-k retrieval metrics."""

    top1_accuracy: float
    top5_accuracy: float


class FailedRetrieval(TypedDict):
    """One failed retrieval example."""

    query: str
    correct_doc_id: int
    correct_title: str
    retrieved_doc_id: int
    retrieved_title: str


class RetrievalRunSummary(TypedDict):
    """Final result returned by the retrieval workflow."""

    top1_accuracy: float
    top5_accuracy: float
    similarity_shape: tuple[int, int]
    failed_queries: list[FailedRetrieval]
    failure_note: str


# Beginner hints for this file:
# - retrieval means "find the closest useful documents for a query"
# - embeddings turn documents and queries into numeric vectors so closeness can be measured
# - top-1 means only the best match; top-5 means the correct answer can appear anywhere in the best five
# - normalize embeddings if you want dot product to behave like cosine similarity
# - always inspect failed queries, because retrieval errors are often very easy to understand by reading them
def describe_retrieval_plan(config: RetrievalConfig) -> RetrievalPlanSummary:
    """Return the main retrieval shapes and metrics."""
    return {
        "document_embeddings": "[num_documents, embedding_dim]",
        "query_embeddings": "[num_queries, embedding_dim]",
        "similarity_matrix": "[num_queries, num_documents]",
        "top_k": config.top_k,
        "metrics": ["top1_accuracy", "top5_accuracy"],
    }

# Return local documents built from the JSON pair dataset.
# Each document should have:
# - `doc_id`
# - `title`
# - `text`
# Each unique document text should get one `doc_id`.
def build_document_set() -> list[RetrievalDocument]:
    """Return documents loaded from the local JSON pair dataset."""
    # Goal:
    # Load the local query/document JSON file and turn its document texts into
    # searchable retrieval documents.
    #
    # Input:
    # - No arguments.
    #
    # Output:
    # - `list[RetrievalDocument]`
    # - Each dictionary must have exactly these keys:
    #   - `"doc_id"`: int, starting at 0
    #   - `"title"`: short title
    #   - `"text"`: searchable document text
    #
    # Example:
    # `{"doc_id": 0, "title": "Document 0", "text": "DataLoader groups Dataset items into batches for training."}`
    #
    # Method:
    # Load this file:
    # `BOOTCAMP_ROOT / "data" / "dual_encoder_retriever_pairs.json"`
    #
    # Use:
    # `with dataset_path.open("r", encoding="utf-8") as file:`
    # `    payload = json.load(file)`
    #
    # The JSON object has a `pairs` key.
    # Each pair has:
    # - `"query"`
    # - `"document"`
    #
    # Steps:
    # 1. Build `dataset_path`.
    # 2. Load the JSON file.
    # 3. Read `pairs = payload["pairs"]`.
    # 4. Walk through the pairs in order.
    # 5. Create one document for each new document text.
    # 6. Use doc ids starting at `0` and counting up by one, with no gaps.
    # 7. Return the document list.
    #
    # Checks:
    # - `documents[0]["doc_id"] == 0`
    # - `set(documents[0]) == {"doc_id", "title", "text"}`
    # - `len(documents) >= 90`
    dataset_path = BOOTCAMP_ROOT / "data" / "dual_encoder_retriever_pairs.json"
    with dataset_path.open("r", encoding="utf-8") as file:
        data_dict = json.load(file)

    pairs = data_dict["pairs"]
    documents = []
    document_id_by_text = {}

    for pair in pairs:
        document_text = pair["document"]
        if document_text in document_id_by_text:
            continue

        doc_id = len(documents)
        document_id_by_text[document_text] = doc_id
        documents.append(
            {
                "doc_id": doc_id,
                "title": f"Document {doc_id}",
                "text": document_text,
            }
        )

    return documents
    #raise NotImplementedError("Return a small local document set.")

# Build the query set from the same JSON pair dataset.
# Each query should include the correct document id.
# In plain words:
# - this gives you the answer key for retrieval accuracy
def build_query_set(documents: list[RetrievalDocument]) -> list[RetrievalQuery]:
    """Return queries with the correct document id."""
    # Goal:
    # Return queries and the id of the document each query should retrieve.
    # This is the answer key for retrieval accuracy.
    #
    # Input:
    # - `documents`: list[RetrievalDocument]
    # - Each document has `doc_id`, `title`, and `text`.
    #
    # Output:
    # - `list[RetrievalQuery]`
    # - Each dictionary must have exactly these keys:
    #   - `"query"`: the search text
    #   - `"correct_doc_id"`: int id of the matching document
    #
    # Example:
    # `{"query": "what is dataloader", "correct_doc_id": 0}`
    #
    # Method:
    # Load the same JSON file:
    # `BOOTCAMP_ROOT / "data" / "dual_encoder_retriever_pairs.json"`
    #
    # Match each pair's `"document"` text to the correct document id from
    # the `documents` list.
    #
    # Steps:
    # 1. Build `document_id_by_text = {document["text"]: document["doc_id"] for document in documents}`.
    # 2. Load the JSON file.
    # 3. Read `pairs = payload["pairs"]`.
    # 4. For each pair, read `pair["query"]`.
    # 5. Look up the id for `pair["document"]`.
    # 6. Return one query dictionary per pair.
    #
    # Checks:
    # - `set(queries[0]) == {"query", "correct_doc_id"}`
    # - Every `correct_doc_id` should exist in the document set.
    # - `len(queries) >= 500`
    document_id_by_text = {document["text"]: document["doc_id"] for document in documents}

    dataset_path = BOOTCAMP_ROOT / "data" / "dual_encoder_retriever_pairs.json"
    with dataset_path.open("r", encoding="utf-8") as file:
        data_dict = json.load(file)

    pairs = data_dict["pairs"]
    queries = []

    for pair in pairs:
        queries.append(
            {
                "query": pair["query"],
                "correct_doc_id": document_id_by_text[pair["document"]],
            }
        )

    return queries
    #raise NotImplementedError("Return queries and matching document ids.")

def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a simple vocabulary for a beginner-friendly embedding baseline."""
    # Goal:
    # Build a word-to-id dictionary from all document and query text.
    #
    # Input:
    # - `texts`: list[str]
    # - Example: `["A tensor stores numbers.", "What stores numbers?"]`
    #
    # Output:
    # - `dict[str, int]`
    # - Example: `{"a": 0, "numbers": 1, "stores": 2, "tensor": 3, "what": 4}`
    #
    # Method:
    # Use a simple teaching tokenizer:
    # `tokens = text.lower().replace(".", "").replace(",", "").split()`
    #
    # This is not a production tokenizer.
    # It is simple so the retrieval math is easier to see.
    #
    # Steps:
    # 1. Create an empty `set` called `tokens`.
    # 2. For each text, lowercase it.
    # 3. Remove basic punctuation such as `.`, `,`, `?`, `!`, `:`, and `;`.
    # 4. Split on spaces.
    # 5. Add non-empty tokens to the set.
    # 6. Sort tokens with `sorted(tokens)` so ids are stable.
    # 7. Return `{token: index for index, token in enumerate(sorted_tokens)}`.
    #
    # Checks:
    # - `isinstance(vocabulary["tensor"], int)` if `"tensor"` appears in the texts.
    # - Running the function twice on the same texts should give the same ids.

    vocab = {}
    unique_words = set()
    for text in texts:
        words = text.lower().split()
        words = [word.strip(string.punctuation) for word in words]
        for word in words:
            if word and word not in unique_words:
                unique_words.add(word)

    sorted_words = sorted(unique_words)
    vocab = {word:i for i, word in enumerate(sorted_words)}

    return vocab
    #raise NotImplementedError("Build a vocabulary from document and query text.")

def encode_texts(texts: list[str], vocabulary: dict[str, int], config: RetrievalConfig) -> torch.Tensor:
    """Encode texts into embedding vectors.
    Simple default path:
    - start with bag-of-words or averaged random word vectors
    - return one vector per text
    """
    # Goal:
    # Return one embedding vector for each text.
    # The output shape must be `[num_texts, config.embedding_dim]`.
    #
    # Input:
    # - `texts`: list[str]
    # - `vocabulary`: dict[str, int]
    # - `config`: RetrievalConfig
    #
    # Output:
    # - `torch.Tensor`
    # - Shape: `[len(texts), config.embedding_dim]`
    #
    # Small example:
    # - `len(texts) = 3`
    # - `config.embedding_dim = 64`
    # - output shape should be `[3, 64]`
    #
    # Method:
    # Simple baseline:
    # use a hashed bag-of-words vector.
    #
    # Formula:
    # - start each text vector as zeros
    # - for each token, find its vocabulary id
    # - map that id into the embedding vector with `bucket = token_id % config.embedding_dim`
    # - add `1.0` to `embedding[bucket]`
    #
    # This can make collisions.
    # A collision means two tokens land in the same bucket.
    # That is fine for this beginner baseline.
    #
    # Steps:
    # 1. Create `embeddings = torch.zeros(len(texts), config.embedding_dim)`.
    # 2. For each text, lowercase and tokenize it the same way as `build_vocabulary`.
    # 3. For each token that is in `vocabulary`, get its id.
    # 4. Compute `bucket = token_id % config.embedding_dim`.
    # 5. Add `1.0` to that bucket for the current row.
    # 6. Return the embedding tensor.
    #
    # Checks:
    # - `embeddings.shape == (len(texts), config.embedding_dim)`
    # - A text with known tokens should not produce an all-zero row.
    embeddings = torch.zeros(len(texts), config.embedding_dim)

    for text_index, text in enumerate(texts):
        words = text.lower().split()
        words = [word.strip(string.punctuation) for word in words]

        for word in words:
            if word not in vocabulary:
                continue

            token_id = vocabulary[word]
            bucket = token_id % config.embedding_dim
            embeddings[text_index, bucket] += 1.0

    return embeddings
    #raise NotImplementedError("Return text embeddings with shape [num_texts, embedding_dim].")

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize embeddings so dot product becomes cosine similarity."""
    # Goal:
    # Return embeddings where each row has length 1.
    # Then dot product behaves like cosine similarity.
    #
    # Input:
    # - `embeddings`: tensor with shape `[num_items, embedding_dim]`
    #
    # Output:
    # - tensor with the same shape as `embeddings`
    #
    # Method:
    # Use:
    # `torch.nn.functional.normalize(embeddings, p=2, dim=1)`
    #
    # Meaning:
    # - `p=2` means normal vector length
    # - `dim=1` means normalize each row
    #
    # Steps:
    # 1. Call `torch.nn.functional.normalize(embeddings, p=2, dim=1)`.
    # 2. Return the normalized tensor.
    #
    # Checks:
    # - `normalized.shape == embeddings.shape`
    # - For non-zero rows, `normalized.norm(dim=1)` should be close to `1`.

    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized
    #raise NotImplementedError("Return normalized embeddings.")

# Compute the full query-document similarity table.
# Return shape:
# - `[num_queries, num_documents]`
# Each row is one query and each column is one document.
def compute_similarity_matrix(query_embeddings: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
    """Return scores with shape `[num_queries, num_documents]`."""
    # Goal:
    # Compare every query embedding with every document embedding.
    #
    # Input:
    # - `query_embeddings`: shape `[num_queries, embedding_dim]`
    # - `document_embeddings`: shape `[num_documents, embedding_dim]`
    #
    # Output:
    # - tensor with shape `[num_queries, num_documents]`
    # - row `i` contains scores for query `i`
    # - column `j` is the score for document `j`
    #
    # Method:
    # Use matrix multiplication:
    # `query_embeddings @ document_embeddings.T`
    #
    # Shape example:
    # `[6, 64] @ [64, 10] -> [6, 10]`
    #
    # Steps:
    # 1. Check both tensors have the same embedding dimension.
    # 2. Compute `similarity_matrix = query_embeddings @ document_embeddings.T`.
    # 3. Return the similarity matrix.
    #
    # Checks:
    # - If query embeddings are `[3, 64]` and document embeddings are `[8, 64]`,
    #   the output should be `[3, 8]`.

    similarity_matrix = query_embeddings @ document_embeddings.T
    return similarity_matrix
    #raise NotImplementedError("Return the query-document similarity matrix.")

def retrieve_top_k(similarity_matrix: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return top-k document indices for each query."""
    # Goal:
    # Return the best document indices for each query.
    # Higher similarity means a better match.
    #
    # Input:
    # - `similarity_matrix`: shape `[num_queries, num_documents]`
    # - `top_k`: int
    #
    # Output:
    # - tensor with shape `[num_queries, top_k]`
    # - each row contains document indices sorted from best to worse
    #
    # Method:
    # Use:
    # `torch.topk(similarity_matrix, k=top_k, dim=1).indices`
    #
    # Meaning:
    # - `dim=1` means choose top documents across each query row
    # - `.indices` gives document ids by column index
    #
    # Steps:
    # 1. Set `safe_k = min(top_k, similarity_matrix.shape[1])`.
    # 2. Call `torch.topk(similarity_matrix, k=safe_k, dim=1)`.
    # 3. Return `.indices`.
    #
    # Checks:
    # - `top_k_indices.shape[0] == similarity_matrix.shape[0]`
    # - `top_k_indices.shape[1] == min(top_k, num_documents)`

    safe_k = min(top_k, similarity_matrix.shape[1])                 # int, at most num_documents
    top_k_result = torch.topk(similarity_matrix, k=safe_k, dim=1)    # shape [num_queries, safe_k]

    return top_k_result.indices                                      # shape [num_queries, safe_k]

    #raise NotImplementedError("Return top-k document ids or indices for each query.")

# Compute top-1 and top-k retrieval accuracy.
# Reading rule:
# - top-1 asks whether the best result was correct
# - top-k asks whether the correct result appeared anywhere in the first k results
def compute_retrieval_metrics(top_k_indices: torch.Tensor, correct_doc_ids: torch.Tensor) -> RetrievalMetricSummary:
    """Return top-1 and top-k retrieval accuracy."""
    # Goal:
    # Compute whether the correct document appears in the retrieved results.
    #
    # Input:
    # - `top_k_indices`: tensor with shape `[num_queries, top_k]`
    # - `correct_doc_ids`: tensor with shape `[num_queries]`
    #
    # Output:
    # Return this exact dictionary:
    # - `"top1_accuracy"`: float
    # - `"top5_accuracy"`: float
    #
    # The returned key is named `top5_accuracy` because `config.top_k` is 5 by default.
    # The local variable below is called `topk_accuracy` because it uses however many columns
    # are inside `top_k_indices`.
    #
    # Method:
    # Top-1:
    # `top_k_indices[:, 0] == correct_doc_ids`
    #
    # Top-k:
    # `(top_k_indices == correct_doc_ids.unsqueeze(1)).any(dim=1)`
    #
    # Steps:
    # 1. Build a boolean tensor for top-1 correctness.
    # 2. Convert it to float and average it.
    # 3. Build a boolean tensor for whether the correct id appears anywhere in the row.
    # 4. Convert it to float and average it.
    # 5. Return both values as Python floats.
    #
    # Checks:
    # - top-1 asks: was the very best result correct?
    # - top-k asks: was the correct result somewhere in the first k results?
    # - top-k accuracy should be greater than or equal to top-1 accuracy.

    top1_correct = top_k_indices[:, 0] == correct_doc_ids
    top1_accuracy = top1_correct.float().mean().item()

    # `top_k_indices` has shape [num_queries, top_k].
    # `correct_doc_ids.unsqueeze(1)` changes its shape from [num_queries] into [num_queries, 1],
    # so each correct id can be compared with every retrieved id in that row.
    # `.any(dim=1)` means: for each query row, did the correct document appear anywhere in top-k?
    topk_correct = (top_k_indices == correct_doc_ids.unsqueeze(1)).any(dim=1)
    topk_accuracy = topk_correct.float().mean().item()

    return {"top1_accuracy": top1_accuracy, "top5_accuracy": topk_accuracy}
    #raise NotImplementedError("Compute top-1 and top-k accuracy.")

def inspect_failed_queries(
    queries: list[RetrievalQuery],
    documents: list[RetrievalDocument],
    top_k_indices: torch.Tensor,
    correct_doc_ids: torch.Tensor,
) -> list[FailedRetrieval]:
    """Return a few failed retrieval examples."""
    # Goal:
    # Return a few cases where top-1 retrieval was wrong.
    # Reading failures helps you understand what the metric missed.
    #
    # Input:
    # - `queries`: list[RetrievalQuery]
    # - `documents`: list[RetrievalDocument]
    # - `top_k_indices`: shape `[num_queries, top_k]`
    # - `correct_doc_ids`: shape `[num_queries]`
    #
    # Output:
    # - `list[FailedRetrieval]`
    # - Each dictionary should have:
    #   - `"query"`
    #   - `"correct_doc_id"`
    #   - `"correct_title"`
    #   - `"retrieved_doc_id"`
    #   - `"retrieved_title"`
    #
    # Method:
    # The top-1 retrieved document is:
    # `retrieved_doc_id = int(top_k_indices[query_index, 0])`
    #
    # Steps:
    # 1. Build `documents_by_id = {document["doc_id"]: document for document in documents}`.
    # 2. Loop over each query index.
    # 3. Read the correct id from `correct_doc_ids`.
    # 4. Read the retrieved id from the first column of `top_k_indices`.
    # 5. If they are different, create one failure dictionary.
    # 6. Keep only the first few failures, for example the first 3.
    # 7. Return the list.
    #
    # Checks:
    # - Every failure should have `retrieved_doc_id != correct_doc_id`.
    # - If there are no failures, return an empty list.
    documents_by_id = {document["doc_id"]: document for document in documents}
    failed_queries: list[FailedRetrieval] = []

    for query_index, query in enumerate(queries):
        correct_doc_id = int(correct_doc_ids[query_index].item())
        retrieved_doc_id = int(top_k_indices[query_index, 0].item())

        if retrieved_doc_id == correct_doc_id:
            continue

        correct_document = documents_by_id[correct_doc_id]
        retrieved_document = documents_by_id[retrieved_doc_id]

        failed_queries.append(
            {
                "query": query["query"],
                "correct_doc_id": correct_doc_id,
                "correct_title": correct_document["title"],
                "retrieved_doc_id": retrieved_doc_id,
                "retrieved_title": retrieved_document["title"],
            }
        )

        if len(failed_queries) == 3:
            break

    return failed_queries
    #raise NotImplementedError("Return failed queries with retrieved and correct documents.")

def write_failure_note(failed_queries: list[FailedRetrieval]) -> str:
    """Write a short note about why retrieval failed."""
    # Goal:
    # Write a short plain-language note about retrieval failures.
    #
    # Input:
    # - `failed_queries`: list[FailedRetrieval]
    #
    # Output:
    # - `str`
    # - Keep it one or two sentences.
    #
    # Method:
    # If there are no failures, say that top-1 retrieval found all correct documents.
    # If there is a failure, describe the first one using the query and titles.
    #
    # Steps:
    # 1. If `failed_queries` is empty, return a short success note.
    # 2. Otherwise, read the first failure.
    # 3. Mention the query.
    # 4. Mention the title retrieved by mistake.
    # 5. Mention the correct title.
    # 6. Return the note.
    #
    # Checks:
    # - The note should be readable out loud.
    # - The note should not be a long log dump.
    if not failed_queries:
        return "Top-1 retrieval found the correct document for every query."

    first_failure = failed_queries[0]
    return (
        f"For the query '{first_failure['query']}', retrieval picked "
        f"'{first_failure['retrieved_title']}' instead of "
        f"'{first_failure['correct_title']}'."
    )
    #raise NotImplementedError("Write a plain-language retrieval failure note.")

def run_retrieval(config: RetrievalConfig) -> RetrievalRunSummary:
    """Run the full retrieval workflow."""
    # Goal:
    # Run the full retrieval workflow and return metrics plus failure analysis.
    #
    # Input:
    # - `config`: RetrievalConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"top1_accuracy"`: float
    # - `"top5_accuracy"`: float
    # - `"similarity_shape"`: tuple[int, int]
    # - `"failed_queries"`: list[FailedRetrieval]
    # - `"failure_note"`: str
    #
    # Method:
    # Use the helper functions in this file in order.
    # Do not rewrite the retrieval logic inside this function.
    #
    # Steps:
    # 1. Build documents with `build_document_set()`.
    # 2. Build queries with `build_query_set(documents)`.
    # 3. Create `document_texts = [document["text"] for document in documents]`.
    # 4. Create `query_texts = [query["query"] for query in queries]`.
    # 5. Build one vocabulary from document and query text:
    #    `build_vocabulary(document_texts + query_texts)`.
    # 6. Encode documents with `encode_texts(document_texts, vocabulary, config)`.
    # 7. Encode queries with `encode_texts(query_texts, vocabulary, config)`.
    # 8. If `config.use_cosine_similarity` is True, normalize both embedding tensors.
    # 9. Compute the similarity matrix.
    # 10. Retrieve top-k document ids.
    # 11. Build `correct_doc_ids` from the query dictionaries.
    # 12. Compute retrieval metrics.
    # 13. Inspect failed queries.
    # 14. Write the failure note.
    # 15. Return the summary dictionary.
    #
    # Checks:
    # - `similarity_shape == (len(queries), len(documents))`
    # - The returned dictionary should pass `check_metrics(metrics)`.
    documents = build_document_set()
    queries = build_query_set(documents)

    document_texts = [document["text"] for document in documents]
    query_texts = [query["query"] for query in queries]

    vocabulary = build_vocabulary(document_texts + query_texts)

    document_embeddings = encode_texts(document_texts, vocabulary, config)
    query_embeddings = encode_texts(query_texts, vocabulary, config)

    if config.use_cosine_similarity:
        document_embeddings = normalize_embeddings(document_embeddings)
        query_embeddings = normalize_embeddings(query_embeddings)

    similarity_matrix = compute_similarity_matrix(query_embeddings, document_embeddings)
    top_k_indices = retrieve_top_k(similarity_matrix, config.top_k)
    correct_doc_ids = torch.tensor([int(query["correct_doc_id"]) for query in queries], dtype=torch.long)

    metrics = compute_retrieval_metrics(top_k_indices, correct_doc_ids)
    failed_queries = inspect_failed_queries(queries, documents, top_k_indices, correct_doc_ids)
    failure_note = write_failure_note(failed_queries)

    print(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"top{config.top_k}_accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"similarity_shape: {tuple(similarity_matrix.shape)}")
    print(f"failure_note: {failure_note}")

    return {
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "similarity_shape": tuple(similarity_matrix.shape),
        "failed_queries": failed_queries,
        "failure_note": failure_note,
    }
    #raise NotImplementedError("Run retrieval and return metrics plus failure analysis.")

def check_metrics(metrics: RetrievalRunSummary) -> bool:
    """Return True when the retrieval summary has the expected keys."""
    required = {"top1_accuracy", "top5_accuracy", "similarity_shape", "failed_queries", "failure_note"}
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
    """Run the retrieval scaffold one step at a time."""
    config = RetrievalConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 12

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_retrieval_plan", describe_retrieval_plan, config)
    if not count_or_stop(ok):
        return
    documents, ok = report_task_progress("build_document_set", build_document_set)
    if not count_or_stop(ok):
        return
    queries, ok = report_task_progress("build_query_set", build_query_set, documents)
    if not count_or_stop(ok):
        return

    document_texts = [document["text"] for document in documents]
    query_texts = [query["query"] for query in queries]
    vocabulary, ok = report_task_progress("build_vocabulary", build_vocabulary, document_texts + query_texts)
    if not count_or_stop(ok):
        return

    document_embeddings, ok = report_task_progress("encode_texts", encode_texts, document_texts, vocabulary, config)
    if not count_or_stop(ok):
        return

    query_embeddings, ok = report_task_progress("encode_texts", encode_texts, query_texts, vocabulary, config)
    if not count_or_stop(ok):
        return

    normalized_documents, ok = report_task_progress("normalize_embeddings", normalize_embeddings, document_embeddings)
    if not count_or_stop(ok):
        return

    normalized_queries, ok = report_task_progress("normalize_embeddings", normalize_embeddings, query_embeddings)
    if not count_or_stop(ok):
        return

    similarity_matrix, ok = report_task_progress("compute_similarity_matrix", compute_similarity_matrix, normalized_queries, normalized_documents)
    if not count_or_stop(ok):
        return

    top_k_indices, ok = report_task_progress("retrieve_top_k", retrieve_top_k, similarity_matrix, config.top_k)
    if not count_or_stop(ok):
        return

    correct_doc_ids = torch.tensor([int(query["correct_doc_id"]) for query in queries], dtype=torch.long)
    _, ok = report_task_progress("compute_retrieval_metrics", compute_retrieval_metrics, top_k_indices, correct_doc_ids)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_retrieval", run_retrieval, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "retrieval summary", float(int(passed)), "all retrieval keys present")
if __name__ == "__main__":
    main()
