import json
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from src.rag_pipeline import QuantumRAGPipeline
from src.embeddings import get_embedder


def compute_similarity(embedder, a: str, b: str) -> float:
    emb = embedder.encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def keyword_score(pred: str, gt: str) -> float:
    gt_words = set(gt.lower().split())
    pred_words = set(pred.lower().split())
    if not gt_words:
        return 0.0
    return len(gt_words & pred_words) / len(gt_words)


def evaluate():
 
    pipeline = QuantumRAGPipeline()
    pipeline.ingest()

    embedder = get_embedder(prefer_sbert=True)

    with open("evaluation/questions.json", "r") as f:
        data = json.load(f)

    results = []

    for item in data:
        q = item["question"]
        gt = item["expected_answer"]

        response = pipeline.generate(q)
        pred = response.answer

        sim = compute_similarity(embedder, pred, gt)
        key_score = keyword_score(pred, gt)

     
        retrieval_hit = any(
            any(word in rc.chunk.text.lower() for word in gt.lower().split())
            for rc in response.retrieved_chunks
        )

        results.append({
            "question": q,
            "similarity": sim,
            "keyword_score": key_score,
            "retrieval_hit": retrieval_hit
        })

        print("\n==============================")
        print(f"Q: {q}")
        print(f"Pred: {pred}")
        print(f"GT: {gt}")
        print(f"Sim: {sim:.3f} | Keyword: {key_score:.3f} | Hit: {retrieval_hit}")


    avg_sim = np.mean([r["similarity"] for r in results])
    avg_key = np.mean([r["keyword_score"] for r in results])
    hit_rate = np.mean([r["retrieval_hit"] for r in results])

    print("\n===== FINAL METRICS =====")
    print(f"Avg Similarity: {avg_sim:.3f}")
    print(f"Avg Keyword Score: {avg_key:.3f}")
    print(f"Retrieval Hit Rate: {hit_rate:.3f}")


if __name__ == "__main__":
    evaluate()