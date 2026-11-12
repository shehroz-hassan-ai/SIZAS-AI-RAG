from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Model sirf ek baar load hota hai."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]