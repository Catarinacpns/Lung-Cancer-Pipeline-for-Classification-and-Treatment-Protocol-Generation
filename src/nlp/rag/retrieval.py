from rank_bm25 import BM25Okapi

def retrieve_top_k_chromadb(query, chromadb_collection, top_k=7):
    """Retrieves top-K most relevant documents using ChromaDB's cosine similarity search."""
    
    # Fetch stored documents
    stored_data = chromadb_collection.get(include=['documents'])

    # Ensure documents exist
    if "documents" not in stored_data or not stored_data["documents"]:
        print("No documents found in ChromaDB.")
        return []

    return stored_data["documents"][:top_k]  # Return top-K documents

def hybrid_retrieval(query, chromadb_collection, top_k=7):
    """Combines BM25 lexical search with ChromaDB semantic search."""

    # Fetch stored documents
    stored_data = chromadb_collection.get(include=['documents'])

    if "documents" not in stored_data or not stored_data["documents"]:
        print("No documents found in ChromaDB.")
        return []

    # Extract documents
    corpus_texts = stored_data["documents"]

    # BM25 Lexical Search
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # Sort by BM25 scores and return top-k documents
    sorted_docs = sorted(zip(corpus_texts, bm25_scores), key=lambda x: x[1], reverse=True)
    
    return [doc[0] for doc in sorted_docs[:top_k]]


def combined_retrieval(query, chromadb_collection, top_k=10):
    """
    Combines BM25 lexical search with ChromaDB semantic search.
    
    - BM25 retrieves keyword-matching documents.
    - ChromaDB retrieves semantically similar documents.
    - The final list merges both results, prioritizing unique and relevant documents.
    """

    # Fetch stored documents
    stored_data = chromadb_collection.get(include=['documents'])

    if "documents" not in stored_data or not stored_data["documents"]:
        print("No documents found in ChromaDB.")
        return []

    # Extract documents
    corpus_texts = stored_data["documents"]

    # BM25 Lexical Search
    tokenized_corpus = [doc.split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # Sort by BM25 scores and return top-k documents
    sorted_docs = sorted(zip(corpus_texts, bm25_scores), key=lambda x: x[1], reverse=True)
    retrieved_bm25_docs = [doc[0] for doc in sorted_docs[:top_k]]

    # ChromaDB Semantic Search
    retrieved_chromadb_docs = retrieve_top_k_chromadb(query, chromadb_collection, top_k=top_k)

    # Combine results (Union of BM25 + ChromaDB), ensuring no duplicates
    combined_docs = list(dict.fromkeys(retrieved_bm25_docs + retrieved_chromadb_docs))  # Maintains order

    # Return only top-k results
    return combined_docs[:top_k]
