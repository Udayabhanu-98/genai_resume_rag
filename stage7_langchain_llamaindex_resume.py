# stage7_final_resume_rag.py

# ==============================
# 1. IMPORTS
# ==============================
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ==============================
# 2. LOAD ENV VARIABLES
# ==============================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# ==============================
# 3. SET EMBEDDING MODEL (LOCAL)
# ==============================
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en"
)

# ==============================
# 4. CREATE DOCUMENTS (RESUME DATA)
# ==============================
docs = [
    Document(text="Strong skills in Python, Machine Learning, SQL"),
    Document(text="Experience with Pandas, NumPy, Scikit-learn"),
    Document(text="Good communication and presentation skills"),
    Document(text="Hands-on experience with AWS cloud services")
]

# ==============================
# 5. BUILD LLAMAINDEX
# ==============================
index = VectorStoreIndex.from_documents(docs)

# Query engine
query_engine = index.as_query_engine()

# ==============================
# 6. LOAD LLM (OPENAI)
# ==============================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==============================
# 7. USER QUERY (JOB DESCRIPTION)
# ==============================
query = """
Evaluate this resume for a Data Scientist role.

Requirements:
- Python
- Machine Learning
- AWS
- Communication skills

Give output in this format:

Match Score: <number>%

Missing Skills:
- skill1
- skill2

Suggestions:
- suggestion1
- suggestion2

Final Summary:
<text>
"""

# ==============================
# 8. RAG STEP (RETRIEVAL)
# ==============================
rag_response = query_engine.query(query)

# Debug (optional)
print("\n🔍 Retrieved Context:\n")
print(rag_response)

# ==============================
# 9. FINAL PROMPT TO LLM
# ==============================
final_prompt = f"""
You are an expert resume evaluator.

Context:
{rag_response}

Task:
{query}
"""

# ==============================
# 10. GENERATE FINAL OUTPUT
# ==============================
response = llm.invoke([
    HumanMessage(content=final_prompt)
])

# ==============================
# 11. OUTPUT
# ==============================
print("\n🚀 FINAL RESPONSE:\n")
print(response.content)