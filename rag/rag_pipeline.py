from openai import OpenAI
from dotenv import load_dotenv
import os

from .chunker import split_text
from .vectorstore import build_faiss_index, search_faiss

# loading api key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# building knowledge base from resume text
def build_resume_kb(resume_text):
    chunks = split_text(resume_text)
    index, _ = build_faiss_index(chunks)
    return {"chunks": chunks, "index": index}

# answering query using kb
def answer_query(query, kb):
    relevant_chunks = search_faiss(query, kb["chunks"], kb["index"], k=4)
    context = "\n\n".join(relevant_chunks)

    prompt = (
        "You are a helpful assistant answering questions about a resume.\n"
        "Use only the information from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # trying primary model
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You answer questions about the given resume."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    # falling back on failure
    except Exception:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You answer questions about the given resume."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
