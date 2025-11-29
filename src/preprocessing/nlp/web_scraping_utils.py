# Standard library imports
import os
import res
import csv
import json
import time
from datetime import datetime

# Other library imports
import requests
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# Configure Selenium WebDriver
def configure_driver():
    options = Options()
    options.headless = True  # Run in headless mode
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--incognito")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--remote-debugging-port=9222")

    driver_path = "/Users/catarinasilva/Desktop/chromedriver-mac-x64/chromedriver"
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver



# Function to parse dates
def try_parse_date(date_str):
    patterns = [
        r"(\w+ \d{1,2}, \d{4})",  # January 29, 2024
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",  # ISO 8601: 2024-10-11T12:00:00Z
        r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
        r"(\d{1,2} \w+ \d{4})"  # 29 January 2024
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            date_text = match.group(1)
            formats = ["%B %d, %Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%d %B %Y"]
            for date_format in formats:
                try:
                    return datetime.strptime(date_text, date_format).strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return "Unknown"



# Function to determine author based on the URL
def determine_author(url):
    domain_author_map = {
        "cancer.org": "American Cancer Society",
        "cancer.gov": "National Cancer Institute",
    }

    domain_match = re.search(r"https?://(www\.)?([a-zA-Z0-9.-]+)/", url)
    if domain_match:
        domain = domain_match.group(2)
        for key in domain_author_map:
            if key in domain:
                return domain_author_map[key]

    return "Unknown"



# Function to extract publication date
def extract_date(driver):
    try:
        raw_date = driver.find_element(By.XPATH, "//meta[@name='pubdate'] | //meta[@property='article:published_time']").get_attribute("content")
        return try_parse_date(raw_date)
    except NoSuchElementException:
        pass

    try:
        time_element = driver.find_element(By.TAG_NAME, "time")
        raw_date = time_element.text.strip() or time_element.get_attribute("datetime")
        return try_parse_date(raw_date)
    except NoSuchElementException:
        pass

    try:
        date_divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'date-reference') or contains(@class, 'date')]")
        for date_div in date_divs:
            raw_date = date_div.text.strip()
            converted_date = try_parse_date(raw_date)
            if converted_date != "Unknown":
                return converted_date
    except NoSuchElementException:
        pass

    return "Unknown"



# Function to extract page content hierarchically
def extract_page_content(driver):
    page_content = []
    current_hierarchy = {"h1": None, "h2": None, "h3": None, "h4": None, "h5": None, "h6": None}

    all_elements = driver.find_elements(By.XPATH, "//h1 | //h2 | //h3 | //h4 | //h5 | //h6 | //p | //li | //img | //figcaption")

    for element in all_elements:
        tag = element.tag_name

        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            current_hierarchy[tag] = element.text.strip()
            for sub_tag in ["h3", "h4", "h5", "h6"]:
                if sub_tag > tag:
                    current_hierarchy[sub_tag] = None
        elif tag in ["p", "li"]:
            heading_path = " > ".join(filter(None, [current_hierarchy["h1"], current_hierarchy["h2"], current_hierarchy["h3"],
                                                    current_hierarchy["h4"], current_hierarchy["h5"], current_hierarchy["h6"]]))
            page_content.append({
                "type": tag,
                "heading_path": heading_path,
                "content": element.text.strip()
            })
        elif tag == "img":
            page_content.append({
                "type": "image",
                "alt_text": element.get_attribute("alt") or "No alt text",
                "title": element.get_attribute("title") or "No title",
                "src": element.get_attribute("src")
            })
        elif tag == "figcaption":
            page_content.append({
                "type": "figcaption",
                "content": element.text.strip()
            })

    return page_content



# Function to extract data from a single page
def extract_data_from_page(driver, url):
    print(f"Scraping: {url}")
    
    driver.get(url)
    
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except TimeoutException:
        print(f"Timeout while waiting for {url}. Skipping...")
        return None

    author, pub_date = determine_author(url), extract_date(driver)
    
    try:
        main_title = driver.find_element(By.TAG_NAME, "h1").text.strip()
    except NoSuchElementException:
        main_title = "No Main Title"

    h2_headings = [element.text.strip() for element in driver.find_elements(By.TAG_NAME, "h2") if element.text.strip()]

    page_content = extract_page_content(driver)

    return {
        "url": url,
        "main_title": main_title,
        "publication_date": pub_date,
        "author": author,
        "h2_headings": "; ".join(h2_headings),
        "content": page_content
    }



def save_to_csv(page_data, output_csv_file):
    headers = ["url", "main_title", "publication_date", "author", "h2_headings"]

    with open(output_csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow({
            "url": page_data["url"],
            "main_title": page_data["main_title"],
            "publication_date": page_data["publication_date"],
            "author": page_data["author"],
            "h2_headings": page_data["h2_headings"]
        })
        


def save_to_json(page_data, output_json_file):
    json_data = {
        "url": page_data["url"],
        "main_title": page_data["main_title"],
        "publication_date": page_data["publication_date"],
        "author": page_data["author"],
        "content": page_data["content"]
    }

    with open(output_json_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)
        


def scrape_and_store_pages(urls, output_directory, output_csv_file):
    driver = configure_driver()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)

    try:
        for i, url in enumerate(urls):
            page_data = extract_data_from_page(driver, url)
            if page_data:
                json_file = os.path.join(output_directory, f"page_{i + 1}.json")
                save_to_json(page_data, json_file)
                save_to_csv(page_data, output_csv_file)
    finally:
        driver.quit()
        
'''
SCRAPPING STATIC WEBSITES 
'''
            
# Function to extract data from a single page
def extract_data_from_page(url):
    print(f"Scraping: {url}")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all headings (h2) and paragraphs (p) in document order
    all_elements = soup.find_all(['h2', 'p'])  # Get all <h2> and <p> tags in order

    page_data = []
    current_heading = None

    # Loop through the elements and structure data
    for element in all_elements:
        if element.name == "h2":  # Heading
            current_heading = element.get_text(strip=True)
        elif element.name == "p" and current_heading:  # Paragraph under the heading
            page_data.append({
                "heading": current_heading,
                "content": element.get_text(strip=True),
                "url": url
            })

    return page_data


# Main function to scrape multiple pages
def scrape_multiple_pages(urls):
    all_data = []

    for url in urls:
        page_data = extract_data_from_page(url)
        all_data.extend(page_data)

    return all_data



# Save data to JSON
def save_to_json(data, output_file):
    # Ensure the output directory exists
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    
    # Save data to a JSON file
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")