import pandas as pd
import pyterrier as pt
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

def get_text_from_index(index_ref, doc_ids: List[str]) -> Dict[str, str]:
    """
    Retrieve document text from the index metadata.

    Args:
        index_ref: PyTerrier index reference
        doc_ids: List of document IDs

    Returns:
        Dict mapping doc_id -> text
    """
    index = pt.IndexFactory.of(index_ref)
    meta_index = index.getMetaIndex()

    result = {}
    for doc_id in doc_ids:
        try:
            # Get internal docid from docno
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                text = meta_index.getItem("text", docid)
                result[doc_id] = text
        except Exception:
            pass  # Document not found

    return result


# Helper for single document lookup
def get_doc_text(index_ref, doc_id: str) -> str:
    """Get text for a single document from the index."""
    texts = get_text_from_index(index_ref, [doc_id])
    return texts.get(doc_id, "")

def find_hard_negatives_bm25(
    query: str,
    positive_doc_ids: set,
    retriever,
    num_negatives: int = 5,
) -> List[str]:
    """
    Find hard negatives using BM25.

    Hard negatives are documents that are lexically similar to the query
    but are NOT the ground truth positive document.

    Args:
        query: The query text
        positive_doc_ids: Set of positive document IDs (to exclude)
        retriever: BM25 retriever
        num_negatives: Number of hard negatives to return

    Returns:
        List of document IDs for hard negatives
    """
    # Implement hard negative mining with BM25
    # Retrieve with BM25
    results = retriever.search(query, num_negatives + len(positive_doc_ids) +20)
    results = results[~results["docno"].isin(positive_doc_ids)]
    hard_negatives = []
    #find hard negatives
    for _, row in results.iterrows():
        hard_negatives.append(row["docno"])
        if len(hard_negatives) == num_negatives:
            break
    return hard_negatives


def create_distillation_dataset(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    index_ref,
    retriever,
    num_hard_negatives: int = 5,
    max_queries: int = None,
) -> List[Dict]:
    """
    Crée un dataset de paires (Query, Document) pour la distillation.
    
    Structure de sortie : Une liste plate où chaque entrée est une paire unique.
    On mélange les vrais positifs (Gold) et les faux positifs (Hard Negatives).
    """
    distillation_pairs = []

    queries_to_process = queries_df.head(max_queries) if max_queries else queries_df

    for _, query_row in tqdm(
        queries_to_process.iterrows(),
        desc="Mining distillation pairs",
        total=len(queries_to_process),
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]


        positive_doc_ids = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())
        
        if not positive_doc_ids:
            continue

        positive_id = list(positive_doc_ids)[0]
        positive_text = get_doc_text(index_ref, positive_id)

        if not positive_text:
            continue
        
        distillation_pairs.append({
            "query": query_text,
            "doc": positive_text,
            "label" : 1.0
            })

        hard_neg_ids = find_hard_negatives_bm25(
            query_text, positive_doc_ids, retriever, num_hard_negatives
        )
        
        hard_neg_texts = get_text_from_index(index_ref, hard_neg_ids)
        
        for nid in hard_neg_ids:
            if nid in hard_neg_texts:
                distillation_pairs.append({
                    "query": query_text,
                    "doc": hard_neg_texts[nid],
                    "label" : 0.0
                })
    print(len(distillation_pairs))
    return distillation_pairs


def create_training_triplets(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    index_ref,
    retriever,
    num_hard_negatives: int = 5,
    max_queries: int = None,
) -> List[Dict]:
    """
    Create training triplets with hard negatives.

    Each triplet contains:
    - query: The question
    - positive: A relevant document
    - negatives: List of hard negative documents

    Args:
        queries_df: DataFrame with queries
        qrels_df: DataFrame with relevance judgments
        index_ref: PyTerrier index reference (for text retrieval)
        retriever: BM25 retriever for hard negative mining
        num_hard_negatives: Number of hard negatives per query
        max_queries: Maximum number of queries to process

    Returns:
        List of training triplets
    """
    triplets = []

    queries_to_process = queries_df.head(max_queries) if max_queries else queries_df

    for _, query_row in tqdm(
        queries_to_process.iterrows(),
        desc="Creating training triplets",
        total=len(queries_to_process),
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        # Get positive documents for this query
        positive_doc_ids = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())

        if not positive_doc_ids:
            continue

        # Get the first positive document
        positive_id = list(positive_doc_ids)[0]
        positive_text = get_doc_text(index_ref, positive_id)

        if not positive_text:
            continue

        # Find hard negatives
        hard_neg_ids = find_hard_negatives_bm25(
            query_text, positive_doc_ids, retriever, num_hard_negatives
        )
        hard_neg_texts = get_text_from_index(index_ref, hard_neg_ids)
        hard_neg_list = [
            hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts
        ]

        if hard_neg_list:
            triplets.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": hard_neg_list,
                }
            )
    
    
    return triplets