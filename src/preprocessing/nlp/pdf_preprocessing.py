# Standard library imports
import os
import re
import csv
import json
import time
from datetime import datetime

# Other library imports
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import pdfplumber


# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extracting from: {pdf_path}")
    
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # Skip empty pages
                extracted_data.append({
                    "page": page_num,
                    "content": text.strip()
                })

    return extracted_data


# Function to save data to a JSON file
def save_to_json(data, output_file):
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")
    

# Function to process multiple PDFs
def process_multiple_pdfs(pdf_paths, output_dir):
    for pdf_path in pdf_paths:
        # Extract the filename without extension to use as the JSON file name
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{file_name}.json")
        
        # Extract text and save to JSON
        pdf_data = extract_text_from_pdf(pdf_path)
        save_to_json(pdf_data, output_file)
        