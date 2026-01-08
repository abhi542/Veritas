import os
import json
import argparse
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss

# Import embedding classes
# Moved inside function to allow optional dependencies

def get_embeddings(provider):
    load_dotenv()
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("langchain-openai is not installed. Please install it to use 'openai' provider.")
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAIEmbeddings(openai_api_key=api_key)
    elif provider == "vertex":
        try:
             from langchain_google_vertexai import VertexAIEmbeddings
        except ImportError:
            raise ImportError("langchain-google-vertexai is not installed. Please install it to use 'vertex' provider.")
            
        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_REGION")
        return VertexAIEmbeddings(project=project, location=location)
    elif provider == "offline":
        print("Using offline embeddings (SentenceTransformers).")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("langchain-huggingface is not installed.")
            
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

def create_and_save_index(docs_file, index_dir, provider="offline"):
    """
    Loads chunks from JSONL, creates embeddings, builds FAISS index, and saves it.
    """
    if not os.path.exists(docs_file):
        print(f"Docs file not found: {docs_file}")
        return

    # Load documents
    documents = []
    print(f"Loading chunks from {docs_file}...")
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Create LangChain Document
            doc = Document(
                page_content=data["text"],
                metadata={
                    "chunk_id": data["chunk_id"],
                    "source": data["source"],
                    "page": data["page"]
                }
            )
            documents.append(doc)
    
    print(f"Loaded {len(documents)} chunks.")
    if not documents:
        print("No documents to index.")
        return

    # Initialize Embeddings
    embedding_model = get_embeddings(provider)

    # Build FAISS index
    print("Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Save index
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        
    vectorstore.save_local(index_dir)
    print(f"FAISS index saved to {index_dir}")

    # We also save the metadata separately if needed, but FAISS in LangChain stores the docstore (metadata) 
    # inside the index directory (index.pkl), so we are good.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and build FAISS index.")
    parser.add_argument("--docs_file", default="data/docs_metadata.jsonl", help="Input JSONL file with chunks")
    parser.add_argument("--index_dir", default="data/faiss_index", help="Output directory for FAISS index")
    parser.add_argument("--provider", default="offline", choices=["openai", "vertex", "offline"], help="Embedding provider")
    args = parser.parse_args()

    # Allow overriding provider via env var if not specified via CLI (or typical priority CLI > Env)
    # Here CLI takes precedence, default is offline.
    # Check .env for default if valid
    load_dotenv()
    env_provider = os.getenv("EMBEDDING_PROVIDER")
    if env_provider and args.provider == "offline": # If user didn't specify arg (defaulted) but env has it different
         # Ideally we'd want to distinguish "defaulted" vs "explicitly set", 
         # but for simplicity let's stick to args or just trust the user passes the right flag.
         # We'll rely on the arg default.
         pass

    create_and_save_index(args.docs_file, args.index_dir, args.provider)
