import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# Reuse embedding logic
from embed_store import get_embeddings

def test_retrieval(index_dir, query, top_k=3, provider="offline"):
    """
    Loads FAISS index and performs a search.
    """
    if not os.path.exists(index_dir):
        print(f"Index directory not found: {index_dir}")
        return

    # Load Embeddings
    try:
        embedding_model = get_embeddings(provider)
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return

    # Load FAISS index
    print(f"Loading FAISS index from {index_dir}...")
    try:
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Search
    print(f"Query: {query}")
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    print(f"\nTop {top_k} Results:")
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Text: {doc.page_content[:200]}...") # Print first 200 chars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test retrieval from FAISS index.")
    parser.add_argument("--index_dir", default="data/faiss_index", help="Directory of FAISS index")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--k", type=int, default=3, help="Number of results")
    parser.add_argument("--provider", default="offline", choices=["openai", "vertex", "offline"], help="Embedding provider")
    args = parser.parse_args()
    
    # Check .env for default provider if needed, similar logic to embed_store
    load_dotenv()
    
    test_retrieval(args.index_dir, args.query, args.k, args.provider)
