import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in the .env file.")

# Initialize Gemini
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)
