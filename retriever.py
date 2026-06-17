def search(collection, query: str, n_results: int = 5):
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    context_chunks = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        context_chunks.append(
            {
                "text": doc,
                "source": f"{source}, page {page}",
            }
        )

    return context_chunks