import os
import uuid
import json
from tqdm import tqdm

import tiktoken

import chromadb

from src.nlp.rag.chunking import chunk_text, chunk_text_openai
from src.nlp.rag.embeddings import get_embedding
from src.nlp.rag.embeddings_openai import get_embedding_openai
from src.nlp.rag.embeddings_utils import structure_documents
from src.nlp.rag.embeddings_openai import structure_documents_openai

def store_embeddings_in_chroma(embeddings, text_chunks, source_file, model_name, chunk_size, overlap):
    if len(embeddings) != len(text_chunks):
        print(f"Warning: Mismatch in embeddings and text chunks for {source_file}")
        return  

    db = db_gemini if model_name == "gemini" else db_minilm  # Use correct DB instance

    for i, emb in enumerate(embeddings):  
        try:
            db.add_texts(
                texts=[text_chunks[i]],
                metadatas=[{
                    "source": source_file,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "model": model_name
                }],
                ids=[f"{source_file}_{model_name}_{chunk_size}_{overlap}_{i}"]
            )
        except Exception as e:
            print(f"Error storing embedding in ChromaDB: {e}")
            
            
def process_and_store_embeddings():
    structured_texts = structure_documents(DATA_FOLDER)

    if not structured_texts:
        print("No structured documents found. Exiting.")
        return

    with tqdm(total=len(structured_texts), desc="Processing Documents", dynamic_ncols=True) as progress_bar:
        for i, doc in enumerate(structured_texts):
            source_file = f"structured_doc_{i}"
            text_data = doc["text"]

            for model_name in EMBEDDING_MODELS.keys():
                chunk_sizes = CHUNK_SIZES["gemini"] if model_name == "gemini" else CHUNK_SIZES["local"]

                for chunk_size in chunk_sizes:
                    for overlap in CHUNK_OVERLAPS:
                        text_chunks = chunk_text(text_data, max_tokens=chunk_size, overlap=overlap)
                        embeddings = get_embedding(text_chunks, model_name)
                        store_embeddings_in_chroma(embeddings, text_chunks, source_file, model_name, chunk_size, overlap)

            progress_bar.update(1)
            
            
def store_embeddings_in_chroma_openai(embeddings, text_chunks, source_file, model_name, chunk_size, overlap):
    """Ensures OpenAI embeddings are stored correctly in ChromaDB with unique IDs."""

    if len(embeddings) != len(text_chunks):
        print(f"Mismatch between embeddings and text chunks for {source_file} ({model_name})")
        return  

    try:
        unique_ids = [str(uuid.uuid4()) for _ in text_chunks]  # Generate unique IDs
        
        db._collection.add(
            documents=text_chunks,  # Store text chunks
            embeddings=embeddings,  # Store embeddings
            metadatas=[{
                "source": source_file,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "model": model_name
            }] * len(text_chunks),
            ids=unique_ids  # Use unique IDs to prevent overwriting
        )

    except Exception as e:
        print(f"Error storing embeddings in ChromaDB: {e}")
        
def process_and_store_embeddings_openai():
    structured_texts = structure_documents_openai(DATA_FOLDER)

    if not structured_texts:
        print("No structured documents found. Exiting.")
        return

    with tqdm(total=len(structured_texts), desc="Processing Documents", dynamic_ncols=True) as progress_bar:
        i = 0
        while i < len(structured_texts):
            batch_texts = []
            batch_sources = []
            while i < len(structured_texts):
                doc = structured_texts[i]
                file_size = len(doc["text"].encode("utf-8"))

                if file_size < 50_000:
                    batch_size = 8
                elif file_size < 200_000:
                    batch_size = 5
                else:
                    batch_size = 2

                if len(batch_texts) >= batch_size:
                    break

                batch_texts.append(doc["text"])
                batch_sources.append(f"structured_doc_{i}")
                i += 1

            for chunk_size in CHUNK_SIZES["openai"]:
                for overlap in CHUNK_OVERLAPS:
                    #print(f"Processing chunk_size={chunk_size}, overlap={overlap}")

                    text_chunks = [chunk for text in batch_texts for key, chunks in chunk_text_openai(text, [chunk_size], [overlap]).items() for chunk in chunks]
                    #print(f"Total Chunks to Embed: {len(text_chunks)}")

                    embeddings = get_embedding_openai(text_chunks)
                    if embeddings:
                        store_embeddings_in_chroma_openai(embeddings, text_chunks, "structured_doc", "text-embedding-ada-002", chunk_size, overlap)

            progress_bar.update(len(batch_texts))