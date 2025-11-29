import os
import json
import time
import random

import tiktoken
import openai
from openai import AzureOpenAI

# Your own imported functions & configured embedding client
from src.nlp.api.gpt4omini import enforce_rate_limits_openai
from src.nlp.api.config import embedding_function

         
def structure_documents_openai(directory):
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


def get_embedding_openai(texts, max_retries=10, batch_size=6):  
    base_delay = 5  
    failed_texts = list(texts)  
    embeddings = {}
    failed_attempts = {text: 0 for text in texts}  

    batch = []
    batch_token_count = 0  

    while failed_texts:
        enforce_rate_limits_openai(sum(len(text.split()) for text in failed_texts))

        try:
            print(f"Requesting embeddings for {len(failed_texts)} chunks")

            batch_embeddings = []
            for chunk in failed_texts:
                chunk_tokens = len(enc.encode(chunk))

                if batch_token_count + chunk_tokens > MAX_TOKENS_PER_MINUTE or len(batch) >= batch_size:
                    #print(f"Sending batch of {len(batch)} chunks | Tokens: {batch_token_count}")
                    new_embeddings = embedding_function.embed_documents(batch)
                    batch_embeddings.extend(new_embeddings)
                    batch = []
                    batch_token_count = 0  

                batch.append(chunk)
                batch_token_count += chunk_tokens

            if batch:
                print(f"Sending final batch of {len(batch)} chunks | Tokens: {batch_token_count}")
                new_embeddings = embedding_function.embed_documents(batch)
                batch_embeddings.extend(new_embeddings)

            for text, embedding in zip(failed_texts, batch_embeddings):
                embeddings[text] = embedding  

            print(f"Successfully got embeddings for {len(failed_texts)} chunks")  
            return [embeddings.get(text, []) for text in texts]  

        except openai.RateLimitError as e:
            delay = min(60, base_delay * (2 ** min(failed_attempts[failed_texts[0]], 6)) + random.uniform(0, 3))
            print(f"Rate limit hit. Retrying in {delay:.2f} seconds.")
            time.sleep(delay)

        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            for text in failed_texts:
                failed_attempts[text] += 1
            failed_texts = [text for text in failed_texts if failed_attempts[text] < max_retries]

    return [embeddings.get(text, []) for text in texts]  
