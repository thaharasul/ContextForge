import re
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
CANDIDATE_K = 30


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_collection_data(collection):
    data = collection.get(
        include=["documents", "metadatas"]
    )

    ids = data.get("ids", [])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    records = []

    for i, document in enumerate(documents):
        if not document:
            continue

        metadata = (
            metadatas[i]
            if i < len(metadatas) and metadatas[i]
            else {}
        )

        records.append({
            "id": ids[i],
            "text": document,
            "metadata": metadata,
        })

    return records


def dense_search(question, collection, top_k):
    count = collection.count()

    if count == 0:
        return []

    model = get_embedding_model()

    embedding = model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0] or []
    documents = result.get("documents", [[]])[0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []

    records = []

    for i, document in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None

        records.append({
            "id": ids[i],
            "text": document,
            "metadata": metadata or {},
            "distance": distance,
        })

    return records


def bm25_search(question, records, top_k):
    if not records:
        return []

    tokenized_corpus = [
        tokenize(record["text"])
        for record in records
    ]

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        record = records[index].copy()
        record["bm25_score"] = float(scores[index])
        results.append(record)

    return results


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K):
    fused = {}

    for ranked_list in ranked_lists:
        for rank, record in enumerate(ranked_list, start=1):
            record_id = record["id"]

            if record_id not in fused:
                fused[record_id] = {
                    **record,
                    "rrf_score": 0.0,
                }

            fused[record_id]["rrf_score"] += 1.0 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def cross_encoder_rerank(question, candidates, top_k):
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [
        (question, record["text"])
        for record in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = []

    for record, score in zip(candidates, scores):
        item = record.copy()
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return ranked[:top_k]


def advanced_retrieve(
    question,
    collection,
    top_k=3,
    method="advanced",
):
    """
    Methods:
      dense    - semantic vector retrieval
      bm25     - keyword retrieval
      hybrid   - dense + BM25 using RRF
      advanced - hybrid RRF + Cross-Encoder reranking
    """

    if collection is None or collection.count() == 0:
        return []

    if method not in {"dense", "bm25", "hybrid", "advanced"}:
        raise ValueError(f"Unknown retrieval method: {method}")

    candidate_k = min(
        collection.count(),
        max(CANDIDATE_K, top_k * 10),
    )

    dense_results = dense_search(
        question,
        collection,
        candidate_k,
    )

    if method == "dense":
        results = dense_results[:top_k]

    else:
        records = load_collection_data(collection)

        keyword_results = bm25_search(
            question,
            records,
            candidate_k,
        )

        if method == "bm25":
            results = keyword_results[:top_k]

        else:
            fused_results = reciprocal_rank_fusion(
                dense_results,
                keyword_results,
            )

            if method == "hybrid":
                results = fused_results[:top_k]
            else:
                results = cross_encoder_rerank(
                    question,
                    fused_results[:candidate_k],
                    top_k,
                )

    # Normalize source metadata for the existing chat interface.
    for record in results:
        metadata = record.get("metadata") or {}

        if not metadata.get("filename"):
            metadata["filename"] = metadata.get(
                "title",
                "Knowledge base document",
            )

        record["metadata"] = metadata

    return results
