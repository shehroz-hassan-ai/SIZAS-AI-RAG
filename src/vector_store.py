import re
import time

from pinecone import Pinecone, ServerlessSpec

from src import config


class VectorStoreError(Exception):
    """Pinecone ka koi bhi masla."""


_pc = None


def _client() -> Pinecone:
    global _pc
    if _pc is None:
        if not config.PINECONE_API_KEY:
            raise VectorStoreError("PINECONE_API_KEY .env mein nahi mili.")
        _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc


def make_namespace(filename: str) -> str:
    """File ke naam se namespace banata hai (1 document = 1 namespace)."""
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower() or "document"


def get_index():
    """Index lata hai, na ho to bana deta hai."""
    try:
        pc = _client()
        if config.INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=config.INDEX_NAME,
                dimension=config.EMBEDDING_DIM,
                metric=config.METRIC,
                spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
            )
            while not pc.describe_index(config.INDEX_NAME).status["ready"]:
                time.sleep(1)
        return pc.Index(config.INDEX_NAME)
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Pinecone connection fail: {e}")


def upsert_chunks(index, namespace: str, chunks: list[dict], vectors: list[list[float]]):
    records = [
        {
            "id": c["chunk_id"],
            "values": v,
            "metadata": {
                "text": c["text"],
                "page": c["page"],
                "doc_name": c["doc_name"],
                "chunk_id": c["chunk_id"],
            },
        }
        for c, v in zip(chunks, vectors)
    ]
    try:
        for i in range(0, len(records), 100):
            index.upsert(vectors=records[i : i + 100], namespace=namespace)
    except Exception as e:
        raise VectorStoreError(f"Upsert fail: {e}")


def query_index(index, vector, namespace: str, top_k: int, metadata_filter=None):
    try:
        res = index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=metadata_filter,
        )
        return res.matches
    except Exception as e:
        raise VectorStoreError(f"Query fail: {e}")


def list_namespaces(index) -> list[str]:
    try:
        stats = index.describe_index_stats()
        return sorted(stats.namespaces.keys())
    except Exception as e:
        raise VectorStoreError(f"Namespaces nahi mile: {e}")


def delete_namespace(index, namespace: str):
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass  # namespace pehle se nahi hai to koi masla nahi