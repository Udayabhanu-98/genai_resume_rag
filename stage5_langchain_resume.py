# stage5_langchain_resume_final.py

# ==============================
# 1. LOAD ENV VARIABLES
# ==============================
from dotenv import load_dotenv
import os

load_dotenv()  # loads OPENAI_API_KEY
print("DEBUG API KEY:", os.getenv("OPENAI_API_KEY"))

# ==============================
# 2. IMPORT LIBRARIES
# ==============================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ==============================
# 3. CREATE RESUME DATA
# ==============================

# This simulates resume content
documents = [
    "Strong skills in Python, Machine Learning, SQL",
    "Experience with Pandas, NumPy, Scikit-learn",
    "Good communication and presentation skills",
    "Hands-on experience with AWS cloud services",
    "Worked on deep learning and NLP projects"
]

# ==============================
# 4. CREATE EMBEDDINGS
# ==============================

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ==============================
# 5. CREATE VECTOR DATABASE
# ==============================

db = FAISS.from_texts(documents, embedding)

# Retriever → fetch relevant chunks
retriever = db.as_retriever(search_kwargs={"k": 3})

# ==============================
# 6. CREATE PROMPT TEMPLATE
# ==============================

prompt = ChatPromptTemplate.from_template("""
You are a Resume Optimization Assistant.

Context (Candidate Resume):
{context}

Job Description:
{question}

Tasks:
1. Give Match Score (0-100)
2. Identify Missing Skills
3. Suggest Improvements
4. Provide Improved Resume Summary

Answer in structured format.
""")

# ==============================
# 7. CREATE LLM
# ==============================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==============================
# 8. FORMAT RETRIEVED DOCS
# ==============================

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# ==============================
# 9. BUILD RAG PIPELINE (LCEL)
# ==============================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ==============================
# 10. USER QUERY (JOB DESCRIPTION)
# ==============================

query = """
Looking for a Data Scientist with:
- Python
- Machine Learning
- AWS
- NLP
- Communication skills
"""

# ==============================
# 11. RUN PIPELINE
# ==============================

response = rag_chain.invoke(query)

print("\n===== FINAL RESPONSE =====\n")
print(response.content)