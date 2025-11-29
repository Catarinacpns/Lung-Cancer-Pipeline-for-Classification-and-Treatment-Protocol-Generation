import os
import time
import requests
from collections import deque

import openai
from openai import AzureOpenAI


def get_azure_openai_rate_limits():
    headers = {
        "Content-Type": "application/json",
        "api-key": subscription_key
    }

    data = {
        "messages": [{"role": "system", "content": "Hello"}],
        "max_tokens": 1
    }

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    try:
        # Make a small request to Azure OpenAI to check rate limits
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            headers = response.headers  # Extract response headers

            # Attempt to extract rate limits (if available)
            rate_limits = {
                "Requests Per Minute (RPM)": headers.get("x-ratelimit-limit-requests", "Not Provided"),
                "Remaining Requests": headers.get("x-ratelimit-remaining-requests", "Not Provided"),
                "Tokens Per Minute (TPM)": headers.get("x-ratelimit-limit-tokens", "Not Provided"),
                "Remaining Tokens": headers.get("x-ratelimit-remaining-tokens", "Not Provided"),
            }

            # Print rate limits
            print("\nAzure OpenAI API Rate Limits:")
            for key, value in rate_limits.items():
                print(f"{key}: {value}")

        else:
            print(f"Failed to retrieve rate limits. HTTP Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Azure OpenAI API: {e}")

def enforce_rate_limits_openai(token_count):
    """Dynamically enforce OpenAI rate limits before making a request."""
    global REQUEST_TIMESTAMPS, TOKENS_USED

    current_time = time.time()

    # Remove old timestamps outside the 60-second window
    while REQUEST_TIMESTAMPS and (current_time - REQUEST_TIMESTAMPS[0]) > 60:
        REQUEST_TIMESTAMPS.popleft()

    # Calculate remaining allowance
    requests_remaining = MAX_REQUESTS_PER_MINUTE - len(REQUEST_TIMESTAMPS)
    tokens_remaining = MAX_TOKENS_PER_MINUTE - TOKENS_USED

    # **Fix: Ensure deque is not empty before accessing**
    if requests_remaining <= 0 or tokens_remaining < token_count:
        if REQUEST_TIMESTAMPS:  # Check if deque is not empty before accessing index 0
            sleep_time = max(1, 60 - (current_time - REQUEST_TIMESTAMPS[0]))
        else:
            sleep_time = 60  # Default sleep time if deque is empty

        print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        TOKENS_USED = 0  # Reset token usage after waiting

    # Update tracking
    REQUEST_TIMESTAMPS.append(current_time)
    TOKENS_USED += token_count

def generate_response_gpt4o(context, query):
    """Generates a response using GPT-4o-mini."""
    openai_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical AI specializing in lung cancer staging and treatment planning."},
            {"role": "user", "content": f"Query: {query}\n\n{context}"}
        ]
    )
    
    return response.choices[0].message.content


