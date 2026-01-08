import argparse
import sys
import os
from src.rag_pipeline import rag

def interactive_mode(provider="offline", llm_provider="openai"):
    print(f"--- RAG System Initialized (Embedding: {provider}, LLM: {llm_provider}) ---")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    # Check for API keys if needed
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found. Generation might fail.")
    
    while True:
        try:
            query = input("\nYour Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("\nThinking...")
            result = rag(query, embedding_provider=provider, llm_provider=llm_provider)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                print("\n[Sources]:")
                for i, (txt, meta) in enumerate(zip(result['retrieved_chunks'], result['retrieved_metadata'])):
                    source = os.path.basename(meta.get('source', 'unknown'))
                    page = meta.get('page', 'N/A')
                    print(f"  {i+1}. {source} (Page {page})")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG system interactively.")
    parser.add_argument("--provider", default="offline", choices=["openai", "vertex", "offline"], help="Embedding provider")
    parser.add_argument("--llm_provider", default="openai", choices=["openai", "vertex", "grok", "gemini"], help="LLM provider")
    args = parser.parse_args()

    # Load env
    from dotenv import load_dotenv
    load_dotenv()
    
    interactive_mode(args.provider, args.llm_provider)
