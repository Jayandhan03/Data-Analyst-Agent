import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
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

# api_key = os.getenv("GROQ_API_KEY")

# if not api_key:
#     raise ValueError("GROQ_API_KEY not found. Make sure it's set in the .env file.")

# # Initialize the Groq Chat Model
# # The 120B model is identified as 'openai/gpt-oss-120b' on the Groq platform.
# llm_model = ChatGroq(
#     model_name="deepseek-r1-distill-llama-70b",
#     groq_api_key=api_key
# )