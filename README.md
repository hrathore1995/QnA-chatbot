# Resume QnA Chatbot (RAG-Based)

A Streamlit application that lets users upload their resume (PDF/DOCX) and ask questions about it. The system uses a Retrieval-Augmented Generation (RAG) pipeline to extract information from the resume and answer queries accurately.

---

## Overview

1. Extracts text from uploaded resume
2. Splits into overlapping text chunks
3. Generates embeddings for all chunks
4. Stores them in a FAISS vector index
5. For each user question: retrieves relevant chunks
6. The OpenAI model answers using only the retrieved context

This ensures accurate, resume-specific responses without hallucination.

---

## Features

* Upload resume in PDF or DOCX format
* Automatic text extraction and cleaning
* Chunking and embedding generation
* FAISS vector store for fast retrieval
* Chatbot-style interface using Streamlit's chat components
* Colored chat bubbles, custom styling, and loading spinner
* Resume preview with stats (characters extracted, chunk count)
* Clear Chat button to reset session

---

## Project Structure

```
nlp_final_project/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
│
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── vectorstore.py
│   └── rag_pipeline.py
```

---

## RAG Pipeline Details

### 1. **Loading**

Extract text from PDF/DOCX using `pdfplumber` and `python-docx`.

### 2. **Chunking**

Split text into overlapping sections (default: 800 characters with 200 overlap).

### 3. **Embedding**

Convert all chunks into numerical vectors using OpenAI embeddings.

### 4. **Vector Indexing**

Store vectors in a FAISS index for similarity search.

### 5. **Retrieval**

For each query:

* embed the question
* retrieve top matching chunks
* apply hybrid scoring (vector similarity + keyword overlap)

### 6. **LLM Answering**

Send the retrieved chunks + question to OpenAI's GPT model.
If `gpt-4.1-mini` fails, fallback to `gpt-4.1`.

---

## Running the App Locally

### 1. Create virtual env

```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add OpenAI API key

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

### 4. Run the app

```
streamlit run app.py
```

---

## Deployment on Streamlit Cloud

1. Push project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Deploy your repository
4. Add OpenAI API key under:

   * **App Settings → Secrets**

```
OPENAI_API_KEY="your_key_here"
```

Streamlit automatically handles sessions for multiple simultaneous users.

---

## requirements.txt

```
streamlit
openai>=1.0.0
faiss-cpu
langchain
langchain-community
python-dotenv
pdfplumber
python-docx
tiktoken
numpy
```

---

## .gitignore

```
.venv/
__pycache__/
.DS_Store
.env
.streamlit/secrets.toml
faiss_index/
```

---

## Acknowledgements

* OpenAI API for embeddings and LLM
* FAISS for vector similarity search
* Streamlit for UI

This project demonstrates a complete end-to-end RAG pipeline integrated into a clean, interactive chatbot interface.
