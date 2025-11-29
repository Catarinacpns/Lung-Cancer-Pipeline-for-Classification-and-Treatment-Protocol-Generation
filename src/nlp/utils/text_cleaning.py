import os
import json
import re
from collections import defaultdict

from bs4 import BeautifulSoup  # For cleaning HTML content


def clean_text(text):
    """Cleans text by removing HTML tags, extra spaces, and special characters."""
    if not text or not isinstance(text, str) or len(text.strip()) < 10:  # Skip empty or very short texts
        return None
    # Check if text looks like HTML before passing to BeautifulSoup
    if "<" in text and ">" in text:  
        text = BeautifulSoup(text, "lxml").get_text()  # Remove HTML tags
    
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text if len(text.split()) > 3 else None  # Skip very short texts


def is_navigation_item(text):
    """Determines if text is a navigation menu item (e.g., Home, About Cancer)."""
    if not text or not isinstance(text, str):  # Ensure text is valid
        return False  
    nav_keywords = {"home", "about", "research", "news", "events", "grants", "training", "contact"}
    return text.lower().strip() in nav_keywords

def clean_website_data(raw_data, filename):
    """Processes and cleans website data stored as a dictionary."""
    
    if not isinstance(raw_data, dict):  # Ensure the root structure is a dictionary
        print(f"Skipping '{filename}': Expected a dictionary but found {type(raw_data).__name__}")
        return None

    cleaned_data = []
    
    # Extract metadata
    metadata = {
        "url": raw_data.get("url", ""),
        "title": raw_data.get("main_title", ""),
        "date": raw_data.get("publication_date", ""),
        "author": raw_data.get("author", "Unknown")
    }

    grouped_content = defaultdict(list)

    # Process content items
    for item in raw_data.get("content", []):
        if not isinstance(item, dict):  # Ensure content items are dictionaries
            continue

        content_type = item.get("type")
        content = clean_text(item.get("content", ""))
        heading = item.get("heading_path", "Uncategorized")

        # Ignore images
        if content_type == "image":
            continue

        # Ignore navigation menu items
        if content_type == "li" and is_navigation_item(content):
            continue

        # Add content to its respective heading
        if content:
            grouped_content[heading].append(content)

    # Structure cleaned data
    for heading, contents in grouped_content.items():
        structured_text = f"## {heading}\n\n" + "\n".join(contents)
        cleaned_data.append({
            "metadata": metadata,
            "heading": heading,
            "content": structured_text
        })

    return cleaned_data if cleaned_data else None  # Return None if no valid content

def clean_pdf_data(raw_data, filename):
    """Processes and cleans extracted text from PDFs."""
    
    if not isinstance(raw_data, list):  # Ensure PDFs have the correct list format
        print(f"Skipping '{filename}': Expected a list but found {type(raw_data).__name__}")
        return None

    cleaned_pages = []

    for page_data in raw_data:
        if not isinstance(page_data, dict):  # Ensure each page entry is a dictionary
            continue

        page_number = page_data.get("page", None)
        content = clean_text(page_data.get("content", ""))

        if content:
            cleaned_pages.append(f"### Page {page_number}\n\n{content}")

    # Combine all pages into one structured document
    cleaned_text = "\n\n".join(cleaned_pages)

    return {"source": filename, "content": cleaned_text} if cleaned_text else None

def clean_list_based_json(raw_data, filename):
    """Processes and cleans website data stored as a list of dictionaries."""
    
    if not isinstance(raw_data, list):  # Ensure the file is a list
        print(f"Skipping '{filename}': Expected a list but found {type(raw_data).__name__}")
        return None

    grouped_content = defaultdict(list)

    for item in raw_data:
        if not isinstance(item, dict):  # Ensure each entry is a dictionary
            continue
        
        heading = item.get("heading", "Uncategorized")
        content = clean_text(item.get("content", ""))

        if content:
            grouped_content[heading].append(content)

    # Structure cleaned data
    cleaned_data = []
    for heading, contents in grouped_content.items():
        structured_text = f"## {heading}\n\n" + "\n".join(contents)
        cleaned_data.append({
            "heading": heading,
            "content": structured_text
        })

    return cleaned_data if cleaned_data else None

def process_json_files(input_directory, output_directory):
    """Cleans and processes all JSON files in the input directory and saves them to the output directory."""
    
    os.makedirs(output_directory, exist_ok=True)

    json_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in '{input_directory}'.")

    for filename in json_files:
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        try:
            with open(input_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)

            # Detect structure and apply appropriate cleaning function
            cleaned_data = None
            if isinstance(raw_data, dict) and "content" in raw_data:
                cleaned_data = clean_website_data(raw_data, filename)
            elif isinstance(raw_data, list) and all(isinstance(entry, dict) for entry in raw_data):
                if "page" in raw_data[0]:  # If first entry has "page", it's a PDF
                    cleaned_data = clean_pdf_data(raw_data, filename)
                else:  # Otherwise, it's a list-based JSON (e.g., scraped_data_ACS.json)
                    cleaned_data = clean_list_based_json(raw_data, filename)
            else:
                print(f"Skipping '{filename}': Unknown structure")
                continue

            if cleaned_data:
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(cleaned_data, file, indent=4, ensure_ascii=False)
                print(f"Cleaned and saved: {output_path}")
            else:
                print(f"Skipping '{filename}': No valid content after cleaning.")

        except Exception as e:
            print(f"Error processing '{filename}': {e}")