import os
import glob
import json
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(pdf_dir, output_file, chunk_size=1000, chunk_overlap=200):
    """
    Loads PDFs from a directory, chunks them, and saves metadata to a JSONL file.
    """
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files.")
    
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # Save metadtata and content to JSONL
    # We include content here effectively as part of the 'metadata' for simple retrieval later
    # In a production DB, you might separate them.
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            # Create a unique ID for the chunk
            chunk_id = f"chunk_{i}"
            record = {
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page")
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Saved {len(chunks)} chunks to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and chunk PDFs.")
    parser.add_argument("--pdf_dir", default="data/pdfs", help="Directory containing PDFs")
    parser.add_argument("--output_file", default="data/docs_metadata.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    load_and_chunk_pdfs(args.pdf_dir, args.output_file)
