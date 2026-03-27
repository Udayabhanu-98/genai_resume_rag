# stage2_prompt_engineering.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_prompt(resume_text, job_role):
    """
    Advanced prompt with:
    - Role definition
    - Few-shot example
    - Clear instructions
    - Structured output
    """

    return f"""
You are a professional resume optimizer.

Your task:
1. Improve the resume for the given job role
2. Use strong action verbs
3. Add missing skills
4. Keep it ATS-friendly


Resume:
{resume_text}

Role:
{job_role}
"""


def improve_resume(resume_text, job_role, temperature=0.5):
    """
    Calls LLM with advanced prompt engineering
    """

    prompt = create_prompt(resume_text, job_role)

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,   # control creativity
        max_tokens=500,
        top_p=0.9
    )

    return response.choices[0].message.content


# Test block
if __name__ == "__main__":

    resume = """
Worked on fraud detection using ML.
Used Python and Pandas.
"""

    print("\n=== NORMAL OUTPUT ===\n")
    print(improve_resume(resume, "Data Scientist"))

    print("\n=== STRICT OUTPUT (temperature=0) ===\n")
    print(improve_resume(resume, "Data Scientist", temperature=0))

    print("\n=== CREATIVE OUTPUT (temperature=1) ===\n")
    print(improve_resume(resume, "Data Scientist", temperature=1))