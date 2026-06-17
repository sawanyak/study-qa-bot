import chromadb
from chromadb.utils import embedding_functions


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_or_create_collection(chat_id: str):
    """Get an existing chat's collection or create a new one."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=f"chat_{chat_id}",
        embedding_function=get_embedding_function(),
    )
    return collection


def add_chunks_to_collection(collection, chunks, doc_id_prefix: str):
    """Add new chunks to an existing collection without wiping it."""
    if not chunks:
        return 0

    collection.add(
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[f"{doc_id_prefix}_chunk_{i}" for i in range(len(chunks))],
    )
    return len(chunks)


def delete_chat_collection(chat_id: str):
    """Remove a chat's collection entirely (when chat is deleted)."""
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(name=f"chat_{chat_id}")
    except Exception:
        pass