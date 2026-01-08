import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    print("Attempting to invoke gemini-pro...")
    result = llm.invoke("Hello, are you available?")
    print("SUCCESS: Model responded.")
    print(f"Response: {result.content}")
except Exception as e:
    print(f"FAILURE: {e}")
