import json
import os
import argparse
import numpy as np
from rag_pipeline import rag 
# uses get_embeddings and get_llm internally via rag_pipeline

def evaluate_generation(test_file, index_dir, provider="offline", llm_provider="openai"):
    """
    Evaluates generation by checking if expected keywords are present in the LLM's answer.
    """
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Load test queries
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    scores = []
    import time
    print(f"Running generation evaluation on {len(test_data)} queries...")

    for item in test_data:
        query = item.get("query")
        expected_keywords = item.get("expected_keywords", [])
        
        if not query:
            continue
            
        # Run RAG
        result = rag(query, index_dir=index_dir, embedding_provider=provider, llm_provider=llm_provider)
        answer = result.get("answer", "")
        
        # Calculate generation score (Recall of keywords)
        found_count = 0
        for keyword in expected_keywords:
            if keyword.lower() in answer.lower():
                found_count += 1
                
        score = found_count / len(expected_keywords) if expected_keywords else 0
        scores.append(score)
        
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.2f}\n")
        time.sleep(10) # Heavy sleep to avoid RPM limits

    avg_score = np.mean(scores) if scores else 0
    print(f"Average Generation Score: {avg_score:.2f}")

    # Save score for aggregation
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/generation_score.json", "w") as f:
        json.dump({"generation_score": avg_score}, f)

    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/test_generation.json")
    parser.add_argument("--index_dir", default="data/faiss_index")
    parser.add_argument("--provider", default="offline", help="Embedding provider")
    parser.add_argument("--llm_provider", default="openai", help="LLM provider (openai, vertex, grok, gemini)")
    args = parser.parse_args()

    evaluate_generation(args.test_file, args.index_dir, provider=args.provider, llm_provider=args.llm_provider)
