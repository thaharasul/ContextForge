from pathlib import Path
import csv
import time
import math
import statistics

import chromadb
from src.advanced_retrieval import advanced_retrieve

DB_PATH = "db"
TOP_K = 3
QUESTION_COUNT = 20

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("wikipedia")

data = collection.get(include=["documents", "metadatas"])

# Group actual Wikipedia chunks by article title.
articles = {}

for doc_id, document, metadata in zip(
    data["ids"], data["documents"], data["metadatas"]
):
    title = metadata.get("title", "Unknown")
    if title == "Unknown" or not document:
        continue

    articles.setdefault(title, []).append({
        "id": doc_id,
        "text": document,
        "chunk_index": metadata.get("chunk_index", 0),
    })

# Select articles with enough content for evaluation.
eligible = [
    (title, sorted(chunks, key=lambda x: x["chunk_index"]))
    for title, chunks in articles.items()
    if len(chunks) >= 2
]

eligible.sort(key=lambda item: item[0].lower())
selected = eligible[:QUESTION_COUNT]

if len(selected) < QUESTION_COUNT:
    raise SystemExit(
        f"Only {len(selected)} articles have multiple chunks. "
        "Need at least 20."
    )

questions = []

for title, chunks in selected:
    first_text = chunks[0]["text"].strip()
    first_sentence = first_text.split(".")[0].strip()

    if len(first_sentence.split()) < 5:
        first_sentence = first_text[:250]

    questions.append({
        "question": (
            f"Find the Wikipedia passage explaining this topic: "
            f"{title}. {first_sentence}"
        ),
        "title": title,
        "relevant_ids": {chunk["id"] for chunk in chunks},
    })

# Save the benchmark questions and their relevance labels.
with open("benchmark_questions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "article_title", "relevant_chunk_ids"])

    for item in questions:
        writer.writerow([
            item["question"],
            item["title"],
            ";".join(sorted(item["relevant_ids"])),
        ])

def calculate_metrics(retrieved_ids, relevant_ids, k=TOP_K):
    ranked = retrieved_ids[:k]
    hits = [1 if doc_id in relevant_ids else 0 for doc_id in ranked]

    relevant_retrieved = sum(hits)
    recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
    precision = relevant_retrieved / k
    hit_rate = 1.0 if relevant_retrieved else 0.0

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        hit / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "hit_rate": hit_rate,
    }

methods = ["dense", "bm25", "hybrid", "advanced"]
results = []

print(f"\nBenchmarking {len(questions)} questions...")
print("Strategies:", ", ".join(methods))
print("Top-K:", TOP_K)

for method in methods:
    print(f"\nRunning {method.upper()}...")

    for index, item in enumerate(questions, start=1):
        start = time.perf_counter()

        retrieved = advanced_retrieve(
            question=item["question"],
            collection=collection,
            top_k=TOP_K,
            method=method,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [record["id"] for record in retrieved]

        metrics = calculate_metrics(
            retrieved_ids,
            item["relevant_ids"],
        )

        results.append({
            "method": method,
            "question": item["question"],
            "article_title": item["title"],
            "retrieved_ids": ";".join(retrieved_ids),
            "latency_ms": round(latency_ms, 2),
            **metrics,
        })

        print(f"  {index}/{len(questions)} completed", end="\r")

# Detailed results.
fields = [
    "method", "question", "article_title", "retrieved_ids",
    "latency_ms", "recall", "precision", "mrr",
    "ndcg", "hit_rate"
]

with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

# Aggregate results.
summary = []

for method in methods:
    subset = [r for r in results if r["method"] == method]

    row = {"method": method}

    for metric in [
        "recall", "precision", "mrr", "ndcg",
        "hit_rate", "latency_ms"
    ]:
        row[metric] = round(
            statistics.mean(r[metric] for r in subset), 4
        )

    summary.append(row)

with open("benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

# Comparison chart.
try:
    import matplotlib.pyplot as plt

    metric_names = [
        "recall", "precision", "mrr", "ndcg",
        "hit_rate", "latency_ms"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, metric in zip(axes.flat, metric_names):
        ax.bar(
            [row["method"] for row in summary],
            [row[metric] for row in summary],
        )
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("RAG Retrieval Strategy Benchmark")
    fig.tight_layout()
    fig.savefig("retrieval_comparison.png", dpi=160)
    plt.close(fig)

except ImportError:
    print("\nMatplotlib unavailable. CSV files were still generated.")

print("\n\n========== BENCHMARK SUMMARY ==========")

for row in summary:
    print(f"\n{row['method'].upper()}")
    for metric, value in row.items():
        if metric != "method":
            print(f"  {metric}: {value}")

print("\nFILES CREATED:")
print("  benchmark_questions.csv")
print("  benchmark_results.csv")
print("  benchmark_summary.csv")
print("  retrieval_comparison.png (if matplotlib is available)")

print("\nBenchmark completed.")
print("Note: This is an initial title-based relevance benchmark,")
print("not a human-validated semantic evaluation.")
