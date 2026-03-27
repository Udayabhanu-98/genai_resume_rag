# stage3_rag_resume.py

import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ================================
# 1. Setup
# ================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================
# 2. Load External Knowledge (RAG Data)
# ================================
documents = [
    "Data Scientist skills include Python, Machine Learning, SQL, Deep Learning, Statistics",
    "Strong communication and data visualization skills are required for Data Scientist",
    "Experience with Pandas, NumPy, and Scikit-learn is important",
    "Knowledge of cloud platforms like AWS or Azure is a plus"
]

# ================================
# 3. Embedding Model
# ================================
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents → embeddings
doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# ================================
# 4. Vector DB (FAISS)
# ================================
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# ================================
# 5. Retrieval Function (RAG Core)
# ================================
def retrieve_context(query, k=2):
    """
    Retrieve top-k relevant documents based on query
    """
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    return [documents[i] for i in indices[0]]

# ================================
# 6. Prompt Builder (Enhanced)
# ================================
def create_prompt(resume_text, job_role, context):
    """
    Prompt includes:
    - Resume
    - Job role
    - Retrieved context (RAG)
    """

    context_text = "\n".join(context)

    return f"""
You are a professional resume optimizer.

Your task:
1. Improve the resume for the given job role
2. Use strong action verbs
3. Add missing skills
4. Keep it ATS-friendly
5. Use the provided job-related context

Context:
{context_text}

Resume:
{resume_text}

Role:
{job_role}
"""

# ================================
# 7. Main Function (RAG + LLM)
# ================================
def improve_resume_rag(resume_text, job_role, temperature=0.5):
    """
    Full pipeline:
    Query → Retrieval → Prompt → LLM
    """

    # Step 1: Create query
    query = job_role   # can also combine resume + role

    # Step 2: Retrieve context (RAG)
    context = retrieve_context(query)

    # Step 3: Build prompt (IMPORTANT)
    prompt = create_prompt(resume_text, job_role, context)

    messages = [
        {
            "role": "system",
            "content": "You are an expert HR and resume writing assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Step 4: LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )

    return response.choices[0].message.content

# ================================
# 8. Test Block
# ================================
if __name__ == "__main__":

    resume = """
Worked on fraud detection using ML.
Used Python and Pandas.
"""

    print("\n=== RAG OUTPUT ===\n")
    print(improve_resume_rag(resume, "Data Scientist"))

    print("\n=== STRICT OUTPUT ===\n")
    print(improve_resume_rag(resume, "Data Scientist", temperature=0))

    print("\n=== CREATIVE OUTPUT ===\n")
    print(improve_resume_rag(resume, "Data Scientist", temperature=1))