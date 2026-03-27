# stage4_advanced_rag_resume.py

import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 1. Resume + JD Data
# -----------------------------
documents = [
    {"text": "Python, Machine Learning, SQL, Deep Learning", "type": "skills"},
    {"text": "Strong communication and data visualization", "type": "soft_skills"},
    {"text": "Experience with Pandas, NumPy, Scikit-learn", "type": "tools"},
    {"text": "Cloud platforms like AWS or Azure", "type": "cloud"}
]

# -----------------------------
# 2. Embeddings
# -----------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

texts = [doc["text"] for doc in documents]
embeddings = embedder.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# -----------------------------
# 3. Query Rewriting
# -----------------------------
def rewrite_query(query):
    prompt = f"Rewrite this job requirement clearly: {query}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# -----------------------------
# 4. Retrieval
# -----------------------------
def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [documents[i] for i in I[0]]

# -----------------------------
# 5. Reranking
# -----------------------------
def rerank(query, results):
    return sorted(results, key=lambda x: len(x["text"]), reverse=True)

# -----------------------------
# 6. Resume Optimization Prompt
# -----------------------------
def generate_response(context, query):
    prompt = f"""
You are a resume optimization assistant.

Context:
{context}

Job Requirement:
{query}

Tasks:
1. Match score (0-100)
2. Missing skills
3. Resume improvement suggestions
4. Final improved summary

Give structured output.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# -----------------------------
# RUN
# -----------------------------
query = "Looking for Data Scientist with ML, Python, AWS, communication skills"

rewritten = rewrite_query(query)
results = retrieve(rewritten)
results = rerank(query, results)

context = " ".join([r["text"] for r in results])

response = generate_response(context, query)
print(response)