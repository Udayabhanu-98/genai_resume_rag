# stage6_llamaindex_resume_final.py

import os
from dotenv import load_dotenv

# Load your .env file (must contain OPENAI_API_KEY=sk-...)
load_dotenv()

# Check if API key is loaded (optional, remove in production)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# 1. Set embedding model (local, free, no API key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. Set LLM (uses the API key from the environment)
Settings.llm = OpenAI(model="gpt-3.5-turbo")   # or "gpt-4" if you have access

# 3. Create documents (simulating resume content)
docs = [
    Document(text="Strong skills in Python, Machine Learning, SQL"),
    Document(text="Experience with Pandas, NumPy, Scikit-learn"),
    Document(text="Good communication and presentation skills"),
    Document(text="Hands-on experience with AWS cloud services")
]

# 4. Build index
index = VectorStoreIndex.from_documents(docs)

# 5. Create query engine
query_engine = index.as_query_engine()

# 6. Define query
query = """
You are an expert resume evaluator. Given the resume content, evaluate it for a Data Scientist role requiring:
- Python
- Machine Learning
- AWS
- Communication

Provide:
1. A match score (0-100%) with justification.
2. Missing skills (if any) with reasoning.
3. Specific suggestions for improvement.
4. Any strengths that stand out.

Format your answer clearly.
"""

# 7. Execute query and print response
response = query_engine.query(query)
print("\nFINAL RESPONSE:\n")
print(response)