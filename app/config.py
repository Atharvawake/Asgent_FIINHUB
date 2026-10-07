import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    """Initializes Google Gemini Flash model."""
    # Check both standard environment variable names
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is missing from environment variables.")
        
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",  # Updated to an active model
        google_api_key=api_key,
        temperature=0.3
    )