import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter # Import the rate limiter

load_dotenv()

# Get API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in the .env file.")

# Initialize a rate limiter for 10 requests per minute
# This is the crucial addition to manage your API calls
rate_limiter = InMemoryRateLimiter(requests_per_second=10 / 60)

# Define the LLM with the rate limiter included
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0,
    max_retries=3,
    rate_limiter=rate_limiter,
)
