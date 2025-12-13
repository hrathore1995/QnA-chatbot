from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from dotenv import load_dotenv
import os

from .chunker import split_text

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_tfidf_kb(resume_text):
    """
    Build a TF-IDF based knowledge base from resume text.

    Returns a dict with:
        - 'chunks': list of text chunks
        - 'vectorizer': fitted TfidfVectorizer
        - 'chunk_vectors': sparse matrix (n_chunks x vocab_size)
    """
    chunks = split_text(resume_text)
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks)
    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "chunk_vectors": chunk_vectors,
    }

def retrieve_tfidf_chunks(query, kb, k=4):
    """
    Retrieve top-k chunks using cosine similarity in TF-IDF space.
    """
    q_vec = kb["vectorizer"].transform([query])          # (1 x vocab)
    # cosine similarity since vectors are L2-normalized in TfidfVectorizer
    scores = (kb["chunk_vectors"] @ q_vec.T).toarray().ravel()
    top_idx = scores.argsort()[::-1][:k]
    return [kb["chunks"][i] for i in top_idx]

def answer_query_tfidf(query, kb, k=4):
    """
    Baseline QA: TF-IDF retrieval + GPT answer over retrieved chunks.
    """
    relevant_chunks = retrieve_tfidf_chunks(query, kb, k=k)
    context = "\n\n".join(relevant_chunks)

    prompt = (
        "You are a helpful assistant answering questions about a resume.\n"
        "Use only the information from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

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

