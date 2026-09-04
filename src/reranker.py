from sentence_transformers import CrossEncoder

_reranker = None

# def get_reranker():
#     global _reranker
#     if _reranker is None:
#         _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     return _reranker


def rerank_and_merge(question: str, hybrid_result: dict, top_k: int = 5) -> list[dict]:
    """
    Scores every candidate (the graph's synthesized answer + each vector
    chunk) against the question with a cross-encoder, and returns the
    top_k ranked highest to lowest.

    Known shortcut, flagged for later refinement: graph_result is
    smart_query's already-answered text, not raw facts, so it's scored
    as one candidate rather than broken into individual facts like the
    vector chunks are.
    """
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = []

    if hybrid_result.get("graph_result"):
        candidates.append({
            "text": hybrid_result["graph_result"],
            "source": "neo4j_graph",
            "origin": "graph",
        })
    candidates.extend(hybrid_result.get("vector_results", []))

    if not candidates:
        return []

    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    print(f"[rerank] {len(candidates)} candidates -> top {min(top_k, len(ranked))}")
    return ranked[:top_k]