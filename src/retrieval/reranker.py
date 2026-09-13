from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    """
    Load the CrossEncoder only once and reuse it
    for all subsequent reranking requests.
    """
    global _reranker

    if _reranker is None:
        print("[reranker] Loading CrossEncoder model...")

        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("[reranker] CrossEncoder model loaded.")

    return _reranker


def rerank_and_merge(
    question: str,
    hybrid_result: dict,
    top_k: int = 5
) -> list[dict]:
    """
    Reranks graph and vector candidates using a CrossEncoder.

    Each candidate gets:
    - original_rank
    - rerank_score
    - final_rank

    The original vector similarity score is preserved.
    """

    # Reuse the already-loaded model
    reranker = get_reranker()

    candidates = []

    # Add graph result as one candidate
    if hybrid_result.get("graph_result"):
        candidates.append({
            "text": hybrid_result["graph_result"],
            "source": "neo4j_graph",
            "origin": "graph",
        })

    # Add vector search results
    candidates.extend(
        hybrid_result.get("vector_results", [])
    )

    if not candidates:
        return []

    # Remember the original retrieval order
    for original_rank, candidate in enumerate(
        candidates,
        start=1
    ):
        candidate["original_rank"] = original_rank

    # Prepare question-document pairs for CrossEncoder
    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    # Calculate reranking scores
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort highest reranking score first
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True
    )

    # Assign final ranking after reranking
    for final_rank, candidate in enumerate(
        ranked,
        start=1
    ):
        candidate["final_rank"] = final_rank

    print(
        f"[rerank] {len(candidates)} candidates "
        f"-> top {min(top_k, len(ranked))}"
    )

    return ranked[:top_k]