from typing import List, Dict, Any


def calculate_precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Calculates Precision@K: The proportion of retrieved documents that are actually relevant."""
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    intersection = top_k_retrieved.intersection(relevant_set)
    return len(intersection) / k


def calculate_recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Calculates Recall@K: The proportion of relevant documents that were successfully retrieved."""
    if not relevant_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    intersection = top_k_retrieved.intersection(relevant_set)
    return len(intersection) / len(relevant_set)


def calculate_mrr(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """Calculates Mean Reciprocal Rank (MRR): Evaluates the rank position of the FIRST relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_results: List[Dict[str, Any]], relevant_ids: List[int], k: int = 3
) -> Dict[str, float]:
    """Comprehensive evaluation suite for the retrieval pipeline."""
    retrieved_ids = [res["doc_id"] for res in retrieved_results]
    return {
        f"precision@{k}": calculate_precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": calculate_recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": calculate_mrr(retrieved_ids, relevant_ids),
    }