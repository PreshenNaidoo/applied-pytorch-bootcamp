"""
============================================================
EXERCISE 35: TINY RAG RETRIEVE THEN ANSWER
============================================================

WHAT YOU ARE BUILDING
You are building a tiny working RAG pipeline.

RAG means retrieval-augmented generation.
In plain words:
- retrieve useful text first
- then answer using that retrieved text

This exercise connects:
- retrieval from the related exercise
- long-context tradeoffs from the related exercise
- language-model evaluation habits from nearby exercises

WHY THIS EXERCISE EXISTS
Retrieval by itself is useful, but real RAG needs a full path:
question -> retrieve documents -> build prompt -> answer -> evaluate.

This exercise teaches:
- how to build a small document store
- how to retrieve top-k documents
- how to build a grounded answer from retrieved text
- how to compare answer quality with and without retrieval
- how to detect when retrieval found the wrong evidence

DATASET
Use a tiny local document dataset written in this file.

Clear default:
- 8 to 12 short course-note documents
- topics: tensors, dataloaders, CNNs, transformers, LoRA, quantization, retrieval, diffusion
- 6 to 10 questions
- each question has:
  - one short answer
  - one supporting document id

Keep each document short.
Each question should have one known answer and one known supporting document.

INPUTS AND OUTPUTS
Question:
- one string

Documents:
- list of short strings

Retriever output:
- top-k document ids
- similarity scores

Answer output:
- answer string
- supporting document ids

METRICS TO TRACK
Track at least:
- top-1 retrieval accuracy
- answer exact match
- grounded answer success
- failure notes

STEP-BY-STEP TODO
1. Build a tiny document set.
2. Build questions with known answers and support ids.
3. Create simple document embeddings.
4. Create simple query embeddings.
5. Retrieve top-k documents.
6. Build a prompt from the question and retrieved documents.
7. Write a tiny rule-based answerer.
8. Compare answering with retrieval against answering without retrieval.
9. Write a short failure note.

DEFINITION OF DONE
You are done when:
- retrieval returns top-k document ids
- answer generation uses retrieved text
- metrics compare with and without retrieval
- failures are explained in plain language
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again


Document = dict[str, int | str]
QuestionExample = dict[str, str | list[int]]
RetrievedDocument = dict[str, int | str | float]
RAGMetrics = dict[str, float | str]


@dataclass
class TinyRAGConfig:
    """Settings for the tiny RAG exercise."""

    top_k: int = 2
    embedding_dim: int = 64


def normalize_text(text: str) -> list[str]:
    """Return simple lowercase tokens."""
    return re.findall(r"\w+", text.lower())


def build_documents() -> list[Document]:
    """Return a tiny local document set."""
    # Goal:
    # - Build the tiny document store that retrieval will search.
    # - Each document should be short and should teach one course concept.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return `list[Document]`.
    # - Each document must have exactly these keys:
    #   - `"id"`: int, unique document id
    #   - `"title"`: str, short title
    #   - `"text"`: str, short evidence text
    # - Example:
    #   `{"id": 0, "title": "Tensors", "text": "A tensor is a grid of numbers."}`
    # - Return at least 8 documents.
    #
    # Method:
    # - Write the documents by hand in a Python list.
    # - Include topics like tensors, dataloaders, CNNs, transformers, LoRA,
    #   quantization, retrieval, and diffusion.
    # - Put the answer phrase for each question directly in the supporting document text.
    #   This lets the rule-based answerer find the answer from retrieved evidence.
    #
    # Steps:
    # 1. Create a list of dictionaries.
    # 2. Give each document a unique integer id.
    # 3. Add `"title"` and `"text"` strings.
    # 4. Return the list.
    #
    # Checks:
    # - `len(documents) >= 8` should be True.
    # - `set(documents[0])` should be `{"id", "title", "text"}`.
    # - Document ids should be unique.
    #
    raise NotImplementedError("Return documents with id, title, and text fields.")


def build_questions() -> list[QuestionExample]:
    """Return questions with answers and support document ids."""
    # Goal:
    # - Build the evaluation questions for the tiny RAG system.
    # - Each question should have a known answer and a known supporting document id.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return `list[QuestionExample]`.
    # - Each item must have exactly these keys:
    #   - `"question"`: str
    #   - `"answer"`: str
    #   - `"support_ids"`: list[int]
    # - Example:
    #   `{"question": "What is a tensor?", "answer": "a grid of numbers", "support_ids": [0]}`
    # - Return at least 6 questions.
    #
    # Method:
    # - Write the list by hand.
    # - Make each answer a short phrase that appears in the supporting document text.
    # - Keep `support_ids` as a list even when there is only one support document.
    #
    # Steps:
    # 1. Create a list of question dictionaries.
    # 2. Add the exact keys `"question"`, `"answer"`, and `"support_ids"`.
    # 3. Make sure each support id exists in `build_documents()`.
    # 4. Return the list.
    #
    # Checks:
    # - `len(questions) >= 6` should be True.
    # - `isinstance(questions[0]["support_ids"], list)` should be True.
    # - The expected answer should appear in the supporting document text.
    #
    raise NotImplementedError("Return question, answer, and support_ids fields.")


def build_vocabulary(documents: list[Document], questions: list[QuestionExample]) -> dict[str, int]:
    """Build a simple bag-of-words vocabulary."""
    # Goal:
    # - Build word ids for the simple bag-of-words retriever.
    # - The vocabulary says which vector index belongs to each word.
    #
    # Inputs:
    # - `documents`: output from `build_documents()`.
    # - `questions`: output from `build_questions()`.
    # - `normalize_text(text)` returns lowercase word tokens, such as
    #   `"A tensor is a grid"` -> `["a", "tensor", "is", "a", "grid"]`.
    #
    # Output:
    # - Return `dict[str, int]`.
    # - Keys are words.
    # - Values are integer vector positions from `0` to `vocab_size - 1`.
    # - Example: `{"tensor": 0, "grid": 1, "numbers": 2}`.
    #
    # Method:
    # - Collect words from:
    #   - each document `"title"` and `"text"`
    #   - each question `"question"` and `"answer"`
    # - Use `normalize_text(...)` for every string.
    # - Add new words in first-seen order so ids are stable and easy to inspect.
    #
    # Steps:
    # 1. Create an empty dictionary called `vocabulary`.
    # 2. Loop over documents.
    # 3. Add tokens from each document title and text.
    # 4. Loop over questions.
    # 5. Add tokens from each question and answer.
    # 6. Return the dictionary.
    #
    # Checks:
    # - `"tensor" in vocabulary` should be True if you used the tensor example.
    # - `min(vocabulary.values())` should be 0.
    # - `max(vocabulary.values())` should be `len(vocabulary) - 1`.
    #
    raise NotImplementedError("Build vocabulary from documents and questions.")


def embed_text(text: str, vocabulary: dict[str, int]) -> torch.Tensor:
    """Return a simple bag-of-words embedding."""
    # Goal:
    # - Turn one text string into a bag-of-words count vector.
    # - Bag-of-words means word order is ignored; only word counts are stored.
    #
    # Inputs:
    # - `text`: any document or question string.
    # - `vocabulary`: word-to-index dictionary.
    # - Small example:
    #   - vocabulary: `{"tensor": 0, "grid": 1}`
    #   - text: `"tensor tensor grid"`
    #
    # Output:
    # - Return a `torch.float32` tensor with shape `[len(vocabulary)]`.
    # - For the example above, return `tensor([2.0, 1.0])`.
    #
    # Method:
    # - Create the vector with:
    #   `vector = torch.zeros(len(vocabulary), dtype=torch.float32)`
    # - For each token from `normalize_text(text)`, if the token is in the vocabulary:
    #   `vector[vocabulary[token]] += 1.0`
    # - Ignore words that are not in the vocabulary.
    #
    # Steps:
    # 1. Create a zero vector.
    # 2. Tokenize the text with `normalize_text(...)`.
    # 3. Add 1.0 to the matching index for each known token.
    # 4. Return the vector.
    #
    # Checks:
    # - `vector.shape` should be `[len(vocabulary)]`.
    # - `vector.dtype` should be `torch.float32`.
    # - Repeated words should increase the count.
    #
    raise NotImplementedError("Return one vector with counts for words in the text.")


def build_document_matrix(documents: list[Document], vocabulary: dict[str, int]) -> torch.Tensor:
    """Return document embeddings with shape `[num_documents, vocab_size]`."""
    # Goal:
    # - Embed every document and stack the vectors into one matrix.
    #
    # Inputs:
    # - `documents`: output from `build_documents()`.
    # - `vocabulary`: output from `build_vocabulary(...)`.
    #
    # Output:
    # - Return a `torch.float32` tensor with shape `[num_documents, vocab_size]`.
    # - Row `i` should be the embedding for `documents[i]`.
    #
    # Method:
    # - Combine title and text before embedding:
    #   `text = f"{document['title']} {document['text']}"`
    # - Embed each document with:
    #   `embed_text(text, vocabulary)`
    # - Stack all vectors with:
    #   `torch.stack(vectors, dim=0)`
    #
    # Steps:
    # 1. Assert there is at least one document.
    # 2. Create one vector per document.
    # 3. Stack the vectors into a matrix.
    # 4. Return the matrix.
    #
    # Checks:
    # - `doc_matrix.shape[0] == len(documents)` should be True.
    # - `doc_matrix.shape[1] == len(vocabulary)` should be True.
    #
    raise NotImplementedError("Embed all documents into one matrix.")


def retrieve_top_k(
    query: str,
    documents: list[Document],
    doc_matrix: torch.Tensor,
    vocabulary: dict[str, int],
    top_k: int,
) -> list[RetrievedDocument]:
    """Return the top-k retrieved documents and scores."""
    # Goal:
    # - Retrieve the top-k documents that best match the query.
    # - Return both the document fields and the similarity score.
    #
    # Inputs:
    # - `query`: question string.
    # - `documents`: list from `build_documents()`.
    # - `doc_matrix`: tensor with shape `[num_documents, vocab_size]`.
    # - `vocabulary`: word-to-index dictionary.
    # - `top_k`: number of documents to return.
    #
    # Output:
    # - Return `list[RetrievedDocument]` with length `min(top_k, len(documents))`.
    # - Each returned dictionary must have exactly these keys:
    #   - `"id"`: int
    #   - `"title"`: str
    #   - `"text"`: str
    #   - `"score"`: float similarity score
    #
    # Method:
    # - Embed the query:
    #   `query_vector = embed_text(query, vocabulary)`
    # - Use cosine similarity:
    #   `dot = doc_matrix @ query_vector`
    #   `doc_norms = doc_matrix.norm(dim=1).clamp_min(1e-8)`
    #   `query_norm = query_vector.norm().clamp_min(1e-8)`
    #   `scores = dot / (doc_norms * query_norm)`
    # - Use `torch.topk(scores, k=min(top_k, len(documents)))`.
    # - `clamp_min(1e-8)` avoids division by zero when a vector has no known words.
    #
    # Steps:
    # 1. Assert `top_k > 0`.
    # 2. Assert `doc_matrix.shape == (len(documents), len(vocabulary))`.
    # 3. Embed the query.
    # 4. Compute cosine scores.
    # 5. Take the top-k scores and indexes.
    # 6. Build result dictionaries with id, title, text, and score.
    # 7. Return the list.
    #
    # Checks:
    # - `len(results) == min(top_k, len(documents))` should be True.
    # - `set(results[0])` should be `{"id", "title", "text", "score"}`.
    # - Scores should be sorted from highest to lowest.
    #
    raise NotImplementedError("Embed the query, compute similarity, and return top-k document records.")


def build_rag_prompt(question: str, retrieved_docs: list[RetrievedDocument]) -> str:
    """Return a prompt that includes retrieved evidence."""
    # Goal:
    # - Build a prompt that gives the answerer the question and retrieved evidence.
    #
    # Inputs:
    # - `question`: one question string.
    # - `retrieved_docs`: output from `retrieve_top_k(...)`.
    #
    # Output:
    # - Return one string.
    # - It should include:
    #   - the question
    #   - each retrieved document id
    #   - each retrieved document title
    #   - each retrieved document text
    #
    # Method:
    # - Use this shape:
    #   `lines = ["Use the evidence to answer the question.", f"Question: {question}", "Evidence:"]`
    # - For each document, add:
    #   `f"[doc {doc['id']}] {doc['title']}: {doc['text']}"`
    # - Join with:
    #   `"\n".join(lines)`
    #
    # Steps:
    # 1. Start the prompt with a short instruction.
    # 2. Add the question.
    # 3. Add one evidence line per retrieved document.
    # 4. Return the joined string.
    #
    # Checks:
    # - The prompt should contain `"Question:"`.
    # - The prompt should contain `"[doc "` when at least one document was retrieved.
    #
    raise NotImplementedError("Join the question and retrieved document text into one prompt.")


def answer_from_retrieved_text(question: str, retrieved_docs: list[RetrievedDocument]) -> str:
    """Return a tiny grounded answer from retrieved text."""
    # Goal:
    # - Return an answer using only retrieved document text.
    # - This is a rule-based toy answerer, not a language model.
    #
    # Inputs:
    # - `question`: the question being answered.
    # - `retrieved_docs`: output from `retrieve_top_k(...)`.
    # - Supporting document text should contain short answer phrases.
    #
    # Output:
    # - Return a short answer string.
    # - Return `""` if no rule finds an answer.
    #
    # Method:
    # - Keep this simple and transparent:
    #   - normalize the question with `normalize_text(question)`
    #   - search the retrieved document texts for known answer phrases
    # - A clean implementation is to put phrase rules in a list, for example:
    #   `rules = [("tensor", "a grid of numbers"), ("dataloader", "batches data"), ...]`
    # - If a question keyword and an answer phrase both match retrieved evidence, return the phrase.
    # - Do not answer from memory if the answer phrase is not in retrieved evidence.
    #
    # Steps:
    # 1. Combine retrieved document text into one lowercase evidence string.
    # 2. Tokenize the question with `normalize_text(...)`.
    # 3. Loop over clear keyword-to-answer rules.
    # 4. Return the matching answer only if the answer phrase appears in the evidence.
    # 5. Return `""` if nothing matches.
    #
    # Checks:
    # - If the right support document is retrieved, the answer should match the known answer.
    # - If no document is passed in, the function should return `""`.
    #
    raise NotImplementedError("Use simple rules to answer from the retrieved documents.")


def answer_without_retrieval(question: str) -> str:
    """Return a weak baseline answer without documents."""
    # Goal:
    # - Return a weak baseline answer that does not read documents.
    # - This shows why retrieval matters.
    #
    # Inputs:
    # - `question`: one question string.
    #
    # Output:
    # - Return one string.
    # - Use the same output for every question, such as:
    #   `"I do not know from the question alone."`
    #
    # Method:
    # - Ignore the question text and return the fixed baseline string.
    # - This is not meant to be smart. It is a comparison point for RAG.
    #
    # Steps:
    # 1. Accept the question argument.
    # 2. Return the fixed baseline string.
    #
    # Checks:
    # - The answer should not change across questions.
    # - The answer should not include facts from documents.
    #
    raise NotImplementedError("Return a simple no-retrieval baseline answer.")


def evaluate_rag(config: TinyRAGConfig) -> RAGMetrics:
    """Run retrieval, answering, and evaluation."""
    # Goal:
    # - Run the full tiny RAG pipeline.
    # - Compare retrieval-augmented answers with the no-retrieval baseline.
    #
    # Inputs:
    # - `config.top_k`: number of documents to retrieve per question.
    # - `config.embedding_dim` is kept for discussion, but this bag-of-words version uses
    #   `len(vocabulary)` as the actual vector size.
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"retrieval_top1_accuracy"`: float from 0.0 to 1.0
    #   - `"rag_exact_match"`: float from 0.0 to 1.0
    #   - `"no_retrieval_exact_match"`: float from 0.0 to 1.0
    #   - `"grounded_answer_success"`: float from 0.0 to 1.0
    #   - `"num_questions"`: float count, such as `6.0`
    #   - `"failure_note"`: short string from `write_failure_note(...)`
    #
    # Method:
    # - Build the pipeline:
    #   `documents = build_documents()`
    #   `questions = build_questions()`
    #   `vocabulary = build_vocabulary(documents, questions)`
    #   `doc_matrix = build_document_matrix(documents, vocabulary)`
    # - For each question:
    #   `retrieved = retrieve_top_k(question_text, documents, doc_matrix, vocabulary, config.top_k)`
    #   `rag_answer = answer_from_retrieved_text(question_text, retrieved)`
    #   `baseline_answer = answer_without_retrieval(question_text)`
    # - Exact match check:
    #   `rag_answer.strip().lower() == expected_answer.strip().lower()`
    # - Retrieval top-1 check:
    #   `retrieved[0]["id"] in support_ids`
    # - Grounded success means:
    #   - answer is exact match
    #   - at least one retrieved document id is in `support_ids`
    #
    # Steps:
    # 1. Build documents and questions.
    # 2. Build vocabulary.
    # 3. Build document matrix.
    # 4. Create counters for retrieval top-1, RAG exact match, baseline exact match,
    #    and grounded success.
    # 5. Loop over questions.
    # 6. Retrieve documents and answer with retrieval.
    # 7. Answer without retrieval.
    # 8. Update counters.
    # 9. Divide each counter by the number of questions.
    # 10. Build the metrics dictionary.
    # 11. Add `failure_note`.
    # 12. Return metrics.
    #
    # Checks:
    # - `0.0 <= metrics["retrieval_top1_accuracy"] <= 1.0` should be True.
    # - `metrics["rag_exact_match"]` should usually be higher than
    #   `metrics["no_retrieval_exact_match"]`.
    # - `check_metrics(metrics)` should be True.
    #
    raise NotImplementedError("Compare retrieved answers with no-retrieval answers.")


def write_failure_note(results: RAGMetrics) -> str:
    """Return a short note about common RAG failures."""
    # Goal:
    # - Explain what can go wrong in a RAG pipeline.
    # - Keep the note tied to the measured results.
    #
    # Inputs:
    # - `results`: metrics from `evaluate_rag(...)`.
    # - Important keys:
    #   - `"retrieval_top1_accuracy"`
    #   - `"rag_exact_match"`
    #   - `"no_retrieval_exact_match"`
    #   - `"grounded_answer_success"`
    #
    # Output:
    # - Return one short plain-language string.
    # - It should mention:
    #   - retrieval can fail by choosing the wrong document
    #   - answering can fail even when retrieval is right
    #   - grounded answers should be checked against supporting evidence
    #
    # Method:
    # - Use concrete numbers from `results`, for example:
    #   `round(float(results["retrieval_top1_accuracy"]), 2)`
    #   `round(float(results["rag_exact_match"]), 2)`
    # - If retrieval accuracy is low, say the retriever often missed the support document.
    # - If RAG exact match is lower than retrieval accuracy, say the answer rule failed
    #   even when evidence was present.
    #
    # Steps:
    # 1. Read retrieval and answer metrics.
    # 2. Write one or two clear sentences.
    # 3. Include at least one metric value.
    # 4. Return the note.
    #
    # Checks:
    # - The note should contain `"retrieval"` and `"evidence"`.
    # - The note should not claim the answer is grounded unless supporting evidence was retrieved.
    #
    raise NotImplementedError("Explain what can go wrong when retrieval finds weak evidence.")


def check_metrics(metrics: RAGMetrics) -> bool:
    """Return True when the RAG report has the expected fields."""
    return (
        "retrieval_top1_accuracy" in metrics
        and "rag_exact_match" in metrics
        and "no_retrieval_exact_match" in metrics
        and "failure_note" in metrics
    )


def report_task_progress(task_name: str, task_fn: Callable, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the tiny RAG scaffold."""
    config = TinyRAGConfig()
    metrics, ok = report_task_progress("evaluate_rag", evaluate_rag, config)
    if ok and metrics is not None:
        print_pass_or_try_again(check_metrics(metrics), "tiny RAG pipeline", float(int(check_metrics(metrics))), "retrieval and answering returned expected fields")


if __name__ == "__main__":
    main()
