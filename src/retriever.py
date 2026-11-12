from src.embedder import embed_query
from src.vector_store import get_index, query_index


def retrieve(
    query: str,
    namespaces: list[str],
    top_k: int,
    threshold: float,
    page_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Selected documents mein se top-k chunks (threshold ke upar) return karta hai."""
    if not query or not query.strip():
        raise ValueError("Sawal khali hai.")
    if not namespaces:
        raise ValueError("Pehle koi document select karo.")

    vector = embed_query(query.strip())
    index = get_index()

    metadata_filter = None
    if page_range:
        metadata_filter = {"page": {"$gte": page_range[0], "$lte": page_range[1]}}

    results = []
    for ns in namespaces:
        for m in query_index(index, vector, ns, top_k, metadata_filter):
            if m.score >= threshold:
                results.append(
                    {
                        "text": m.metadata["text"],
                        "page": int(m.metadata["page"]),
                        "doc_name": m.metadata["doc_name"],
                        "chunk_id": m.metadata["chunk_id"],
                        "score": float(m.score),
                    }
                )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]