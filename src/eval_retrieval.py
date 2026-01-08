import json
import os
import argparse
import numpy as np
from embed_store import get_embeddings
from langchain_community.vectorstores import FAISS

def evaluate_retrieval(test_file, index_dir, top_k=3, provider="offline"):
    """
    Evaluates retrieval by checking if expected keywords are present in retrieved chunks.
    """
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Load test queries
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Load FAISS
    try:
        embedding_model = get_embeddings(provider)
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    scores = []

    print(f"Running evaluation on {len(test_data)} queries...")

    for item in test_data:
        query = item.get("query")
        expected_keywords = item.get("expected_keywords", [])
        
        if not query or not expected_keywords:
            continue

        # Retrieve
        docs = retriever.invoke(query)
        retrieved_text = " ".join([doc.page_content.lower() for doc in docs])
        
        # Calculate retrieval score (Recall of keywords)
        found_count = 0
        for keyword in expected_keywords:
            if keyword.lower() in retrieved_text:
                found_count += 1
        
        score = found_count / len(expected_keywords) if expected_keywords else 0
        scores.append(score)
        
        print(f"Query: {query}")
        print(f"Retrieved: {retrieved_text[:150]}...")
        print(f"Score: {score:.2f}\n")

    avg_score = np.mean(scores) if scores else 0
    print(f"\nAverage Retrieval Score: {avg_score:.2f}")

    # Save score for aggregation
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/retrieval_score.json", "w") as f:
        json.dump({"retrieval_score": avg_score}, f)
        
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/test_retrieval.json")
    parser.add_argument("--index_dir", default="data/faiss_index")
    parser.add_argument("--provider", default="offline")
    args = parser.parse_args()
    
    # Check .env if needed
    from dotenv import load_dotenv
    load_dotenv()

    evaluate_retrieval(args.test_file, args.index_dir, top_k=3, provider=args.provider)
