from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load your .env file to get HF_API_TOKEN
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

LLM_MODEL_ID = "deepsseek-ai/DeepSeek-V3.2"  # ~685M params, also inference available
  # ~229M params, hosted for inference
 # if available for API (check HF)
  # free model

# Initialize the HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id=LLM_MODEL_ID,
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=HF_API_TOKEN
)

# Test prompt
response = llm.invoke("Explain blackbody radiation in simple terms.")
print("\n--- LLM Response ---\n")
print(response)
