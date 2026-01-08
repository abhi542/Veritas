import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# LCEL Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# LLMs
# Moved inside function
try:
    from src.embed_store import get_embeddings
except ImportError:
    from embed_store import get_embeddings

load_dotenv()

def get_llm(provider):
    if provider == "openai":
        try:
             from langchain_openai import ChatOpenAI
        except ImportError:
             raise ImportError("langchain-openai is not installed.")
             
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    elif provider == "vertex":
        try:
            from langchain_google_vertexai import VertexAI
        except ImportError:
            raise ImportError("langchain-google-vertexai is not installed.")
            
        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_REGION")
        return VertexAI(model_name="gemini-pro", project=project, location=location)
    elif provider == "grok":
        try:
             from langchain_openai import ChatOpenAI
        except ImportError:
             raise ImportError("langchain-openai is not installed.")
             
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in .env")
        
        # Grok uses OpenAI-compatible API
        return ChatOpenAI(
            model="grok-beta", 
            openai_api_key=api_key, 
            openai_api_base="https://api.x.ai/v1"
        )
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai is not installed.")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
            
        return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)
        # return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=api_key)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def rag(query, index_dir="data/faiss_index", top_k=3, embedding_provider="offline", llm_provider="openai"):
    """
    End-to-end RAG function using LCEL. 
    Returns a dictionary with query, answer, and retrieved documents.
    """
    
    # 1. Load Embeddings & Index
    try:
        embedding_model = get_embeddings(embedding_provider)
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return {"error": f"Failed to load index/embeddings: {e}"}

    # 2. Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. Setup LLM
    try:
        llm = get_llm(llm_provider)
    except Exception as e:
        return {"error": f"Failed to initialize LLM: {e}"}

    # 4. Define Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # 5. Define Formatting Helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Create Chain (LCEL)
    # We want to return source documents, so we use RunnableParallel to keep 'context'
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # 7. Run Chain
    # invoke takes the query string directly because of RunnablePassthrough() in "question"
    result = rag_chain_with_source.invoke(query)
    
    # Format output
    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_chunks": [doc.page_content for doc in result["context"]],
        "retrieved_metadata": [doc.metadata for doc in result["context"]]
    }

if __name__ == "__main__":
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--embedding_provider", default="offline")
    parser.add_argument("--llm_provider", default="openai") # Default to OpenAI for now
    args = parser.parse_args()

    res = rag(args.query, embedding_provider=args.embedding_provider, llm_provider=args.llm_provider)
    print(res)
