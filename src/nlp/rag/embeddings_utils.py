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


# === Function to Structure Documents from JSON Files ===
def structure_documents(directory):
    structured_documents = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    print(f"Found {len(json_files)} JSON files.")  # Debug print

    for filename in json_files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Handle Website Data (Stored as List)
        if isinstance(data, list) and all(isinstance(entry, dict) for entry in data) and "heading" in data[0]:
            for entry in data:
                metadata = entry.get("metadata", {})
                structured_text = f"Title: {metadata.get('title', 'Unknown')}\nURL: {metadata.get('url', '')}\nDate: {metadata.get('date', '')}\nAuthor: {metadata.get('author', 'Unknown')}\n\n"
                structured_text += f"## {entry.get('heading', 'Uncategorized')}\n\n{entry.get('content', '')}"
                
                structured_documents.append({
                    "text": structured_text,
                    "metadata": metadata
                })

        # Handle PDF Data (Stored as Single Dict)
        elif isinstance(data, dict) and "content" in data:
            if isinstance(data["content"], str):  # If content is a single string, use it directly
                structured_text = f"## PDF Document: {data.get('source', 'Unknown')}\n\n{data['content']}"
                structured_documents.append({
                    "text": structured_text,
                    "metadata": {"source": data.get("source", "Unknown"), "type": "pdf"}
                })
            elif isinstance(data["content"], list):  # If content is a list, process each item
                grouped_content = []
                for item in data["content"]:
                    if isinstance(item, dict) and "content" in item:  # Ensure each entry is a dictionary
                        content_text = item.get("content", "").strip()
                        if content_text:
                            grouped_content.append(content_text)
                
                structured_text = f"## PDF Document: {data.get('source', 'Unknown')}\n\n" + "\n\n".join(grouped_content)
                structured_documents.append({
                    "text": structured_text,
                    "metadata": {"source": data.get("source", "Unknown"), "type": "pdf"}
                })
        
        else:
            print(f"Skipping {filename}: Unrecognized structure.")
            continue

    print(f"Structured {len(structured_documents)} documents.")  # Debug print
    return structured_documents

            
def load_all_embeddings(db):
    stored_data = db._collection.get(include=['documents', 'embeddings', 'metadatas'])
    return stored_data