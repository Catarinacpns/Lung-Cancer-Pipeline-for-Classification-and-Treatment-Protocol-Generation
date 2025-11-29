import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import google.generativeai as genai


# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

embedding_function = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    azure_deployment=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT"),
    openai_api_version=AZURE_API_VERSION,
    max_retries=50,
)

# Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
