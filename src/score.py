from evaluate import load as load_metric
from tqdm.auto import tqdm
import pandas as pd
from typing import List, Dict, Tuple
def evaluate_generation(ground_truth: List[str], generation: List[str]) -> dict:
    bertscore = load_metric("bertscore")

    results = bertscore.compute(
        predictions=generation,    
        references=ground_truth,   
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli", 


    )
    return {
        "precision": sum(results["precision"]) / len(results["precision"]),
        "recall": sum(results["recall"]) / len(results["recall"]),
        "f1": sum(results["f1"]) / len(results["f1"]),
    }
    

def evaluate_retriever(
    retriever_fn,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    ks: List[int] = [1, 5, 10],
) -> Dict:
    """
    Evaluate a retriever on MRR and Recall@k metrics.

    Args:
        retriever_fn: Function that takes a query and returns [(doc_id, score)]
        queries_df: DataFrame with queries
        qrels_df: DataFrame with relevance judgments
        ks: Values of k for Recall@k

    Returns:
        Dict with metrics
    """
    reciprocal_ranks = []
    recalls = {k: [] for k in ks}

    for _, query_row in tqdm(
        queries_df.iterrows(), desc="Evaluation", total=len(queries_df)
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        # Get relevant docs for this query
        relevant_docs = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())

        if not relevant_docs:
            continue

        # Get results
        results = retriever_fn(query_text)
        retrieved_ids = [doc_id for doc_id, _ in results]

        # MRR
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall@k
        for k in ks:
            top_k = set(retrieved_ids[:k])
            recalls[k].append(len(top_k & relevant_docs) / len(relevant_docs))

    metrics = {
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
    }
    for k in ks:
        metrics[f"Recall@{k}"] = sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0

    return metrics