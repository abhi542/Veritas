import json
import os
import argparse

def aggregate_scores():
    print("\n--- RAG Quality Index (RQI) Report ---\n")
    
    # 1. Load Scores
    try:
        with open("data/results/retrieval_score.json", "r") as f:
            retrieval = json.load(f).get("retrieval_score", 0)
    except FileNotFoundError:
        print("Warning: Retrieval score not found. Run src/eval_retrieval.py first.")
        retrieval = 0

    try:
        with open("data/results/generation_score.json", "r") as f:
            generation = json.load(f).get("generation_score", 0)
    except FileNotFoundError:
        print("Warning: Generation score not found. Run src/eval_generation.py first.")
        generation = 0

    try:
        with open("data/results/ragas_score.json", "r") as f:
            ragas = json.load(f).get("ragas_score", 0)
    except FileNotFoundError:
        print("Warning: RAGAS score not found. Run src/eval_ragas.py first.")
        ragas = 0

    # 2. Weights (Senior Engineer logic from README)
    W_RETRIEVAL = 0.4
    W_GENERATION = 0.3
    W_RAGAS = 0.3

    # 3. Calculate RQI
    rqi = (retrieval * W_RETRIEVAL) + (generation * W_GENERATION) + (ragas * W_RAGAS)

    # 4. Print Report
    print(f"Retrieval Score (Recall):   {retrieval:.2f}  (Weight: {W_RETRIEVAL})")
    print(f"Generation Score (Facts):   {generation:.2f}  (Weight: {W_GENERATION})")
    print(f"RAGAS Score (Reasoning):    {ragas:.2f}  (Weight: {W_RAGAS})")
    print("-" * 40)
    
    # Color code output if terminal supports it (simple logic)
    grade = "F"
    if rqi >= 0.9: grade = "A+"
    elif rqi >= 0.8: grade = "A"
    elif rqi >= 0.7: grade = "B"
    elif rqi >= 0.6: grade = "C"
    elif rqi >= 0.5: grade = "D"

    print(f"Final RQI Score:            {rqi:.2f} / 1.00")
    print(f"System Grade:               {grade}")
    print("-" * 40)
    
    # Interpretation
    if retrieval < 0.5:
        print("\n[CRITICAL] Retrieval is failing. Your index or chunks are poor. Fix this first.")
    elif generation < 0.5:
        print("\n[WARNING] LLM is not using the retrieved context correctly. Check prompts.")
    elif ragas < 0.5:
        print("\n[NOTICE] System answers simplistic questions but fails on reasoning.")
    else:
        print("\n[SUCCESS] System is healthy across all metrics.")

if __name__ == "__main__":
    aggregate_scores()
