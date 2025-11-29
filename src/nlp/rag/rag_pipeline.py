# --- Retrieval Methods (cosine, BM25, hybrid) ---
from src.nlp.rag.retrieval import (
    retrieve_top_k_chromadb,
    hybrid_retrieval,
    combined_retrieval,
)

# --- Structured Prompt Generation (TNM + metadata) ---
from src.nlp.prompt.prompt import generate_structured_prompt_tnm

# --- LLM Text Generation ---
from src.nlp.api.gpt4omini import generate_response_gpt4o
from src.nlp.api.gemini_2flash import generate_response_gemini

def retrieval_and_response_pipeline(
    query, 
    embedding_model, 
    retrieval_method,  # Now supports 'cosine', 'bm25', and 'combined'
    llm_model, 
    t_stage, 
    n_stage, 
    m_stage, 
    histopath_grade, 
    cancer_type, 
    age, 
    gender, 
    additional_info=None, 
    top_k=10  # Increased default to allow better ranking
):
    """Runs the full pipeline: retrieval → structured TNM prompt → response generation."""
    
    # Select ChromaDB collection based on embedding model
    if embedding_model == "gemini":
        chromadb_collection = db_gemini
    elif embedding_model == "minilm":
        chromadb_collection = db_minilm
    elif embedding_model == "openai":
        chromadb_collection = db_openai
    else:
        raise ValueError("Invalid embedding model. Choose from 'gemini', 'minilm', or 'openai'.")

    # Retrieve relevant documents based on the selected method
    if retrieval_method == "cosine":
        retrieved_docs = retrieve_top_k_chromadb(query, chromadb_collection, top_k=top_k)
    elif retrieval_method == "bm25":
        retrieved_docs = hybrid_retrieval(query, chromadb_collection, top_k=top_k)
    elif retrieval_method == "combined":
        retrieved_docs = combined_retrieval(query, chromadb_collection, top_k=top_k)
    else:
        raise ValueError("Invalid retrieval method. Choose from 'cosine', 'bm25', or 'combined'.")

    if not retrieved_docs:
        return "No relevant documents found in ChromaDB."

    # Generate structured TNM staging prompt with patient-specific details
    structured_prompt = generate_structured_prompt_tnm(
        t_stage, n_stage, m_stage, histopath_grade, cancer_type, age, gender, additional_info
    )

    retrieved_context = "\n\n".join(retrieved_docs)

    # Create the final structured prompt
    final_prompt = f"{structured_prompt}\n\n### **Retrieved Guidelines & Literature**\n{retrieved_context}"

    # Generate response using selected LLM
    if llm_model == "gpt-4o":
        response = generate_response_gpt4o(final_prompt, query)
    elif llm_model == "gemini":
        response = generate_response_gemini(final_prompt, query)
    else:
        raise ValueError("Invalid LLM model. Choose from 'gpt-4o' or 'gemini'.")

    return response