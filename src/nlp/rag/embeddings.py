import os
import json
import torch
import time
import tiktoken
import google.generativeai as genai
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings  

## **Gemini Embeddings (768-Dim)**
class GeminiEmbeddings(Embeddings):
    """Embedding function wrapper for Gemini text-embedding-004 (768-Dim)."""
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = genai.embed_content(model="models/text-embedding-004", content=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Gemini API Error: {e}")
                embeddings.append([0] * 768)  # Placeholder for failed embeddings
        return embeddings

    def embed_query(self, text):
        """Returns embedding for a single query."""
        return self.embed_documents([text])[0]  # Single input


## **MiniLM Embeddings (384-Dim)**
class MiniLMEmbeddings(Embeddings):
    """Embedding function wrapper for all-MiniLM-L6-v2 (384-Dim)."""
    def __init__(self, model_name="all-MiniLM-L6-v2", device="mps"):
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_documents(self, texts):
        try:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            print(f"MiniLM Embedding Error: {e}")
            return [[0] * 384] * len(texts)  # Placeholder for failed embeddings

    def embed_query(self, text):
        """Returns embedding for a single query."""
        return self.embed_documents([text])[0]  # Single input
    
def get_embedding(texts, model_name, batch_size=5):  
    if model_name == "gemini":
        embeddings = []
        for text in texts:
            try:
                response = genai.embed_content(model="models/text-embedding-004", content=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Gemini API Error: {e}")
                embeddings.append([0] * 768)  # Placeholder for failed embeddings
        return embeddings

    else:  # Local Models (MiniLM)
        model_path = EMBEDDING_MODELS[model_name]
        model = SentenceTransformer(model_path).to(device)  

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    batch_embeddings = model.encode(batch, convert_to_numpy=True).tolist()
                except Exception as e:
                    print(f"Local Model Error: {e}")
                    batch_embeddings = [[0] * 384] * len(batch)  # Placeholder for failed embeddings
                embeddings.extend(batch_embeddings)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return embeddings