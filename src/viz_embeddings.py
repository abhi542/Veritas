import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# We need to access FAISS index vectors directly or reconstruction from docstore if possible.
# LangChain FAISS wrapper doesn't easily expose all vectors at once without some hacking or just reloading them from the embedding model.
# Since we saved metadata in docs_metadata.jsonl, we can re-embed them or try to access the FAISS index directly.
# Accessing FAISS index directly is faster if we can.

from langchain_community.vectorstores import FAISS
from embed_store import get_embeddings

def visualize_embeddings(index_dir, provider="offline", output_file="data/embeddings_visual.png"):
    """
    Visualizes embeddings using t-SNE.
    """
    if not os.path.exists(index_dir):
        print(f"Index directory not found: {index_dir}")
        return

    # Load Embeddings & Index
    try:
        embedding_model = get_embeddings(provider)
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Extract vectors from FAISS index
    index = vectorstore.index
    ntotal = index.ntotal
    print(f"Index contains {ntotal} vectors.")
    
    if ntotal == 0:
        print("No vectors to visualize.")
        return

    # Reconstruct vectors (if index supports it, e.g. Flat or IVF with reconstruction)
    # IndexFlatL2 supports reconstruction.
    try:
        vectors = index.reconstruct_n(0, ntotal)
    except Exception as e:
        print(f"Could not reconstruct vectors directly from index (might be HNSW/IVF without full storage): {e}")
        print("Attempting to rely on docstore IDs if available...")
        # Fallback: We might need to handle this differently if reconstruction fails.
        return

    # Create DataFrame for plotting
    # We need metadata to label points.
    # LangChain stores docstore in memory.
    docstore = vectorstore.docstore
    # The mapping from index id to docstore id is in index_to_docstore_id
    index_to_docstore_id = vectorstore.index_to_docstore_id
    
    sources = []
    
    for i in range(ntotal):
        doc_id = index_to_docstore_id.get(i)
        if doc_id:
            doc = docstore.search(doc_id)
            if isinstance(doc, (str, dict)): # Sometimes it might return raw dict depending on version
               # LangChain InMemoryDocstore returns Document object usually
               pass
            
            # extract source
            if hasattr(doc, "metadata"):
               sources.append(doc.metadata.get("source", "unknown"))
            else:
               sources.append("unknown")
        else:
            sources.append("unknown")

    # Clean up sources - maybe just filename
    sources = [os.path.basename(s) if s else "unknown" for s in sources]

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, ntotal-1))
    embeddings_2d = tsne.fit_transform(vectors)

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "source": sources
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="source", palette="viridis", s=100)
    plt.title("RAG Embeddings Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="data/faiss_index")
    parser.add_argument("--provider", default="offline")
    parser.add_argument("--output_file", default="data/embeddings_visual.png")
    args = parser.parse_args()

    # Check .env
    from dotenv import load_dotenv
    load_dotenv()
    
    visualize_embeddings(args.index_dir, args.provider, args.output_file)
