# stage1_basic_llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def improve_resume(resume_text, job_role):
    prompt = f"""
You are an expert resume writer. Improve the following resume for the role: {job_role}

Resume:
{resume_text}

Give:
- Improved summary
- Better bullet points
- Missing skills
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Test it
if __name__ == "__main__":
    resume = "I worked on machine learning projects and used Python."
    result = improve_resume(resume, "Data Scientist")
    print(result)