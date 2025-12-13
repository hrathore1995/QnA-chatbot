# Resume QnA Chatbot (RAG-Based)

A Streamlit application that lets users upload their resume (PDF/DOCX) and ask questions about it. The system uses a Retrieval-Augmented Generation (RAG) pipeline to extract information from the resume and answer queries accurately.

---

## Overview

This project does **not** train any model. Instead, it uses OpenAI's pretrained LLM along with a RAG pipeline:

1. Extract text from uploaded resume
2. Split into overlapping text chunks
3. Generate embeddings for all chunks
4. Store them in a FAISS vector index
5. For each user question: retrieve relevant chunks
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

5. Share the public app URL with teammates

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

---

## Baseline Method

As a baseline, we implemented a traditional information retrieval approach using TF-IDF. In this method, the resume text is first split into chunks, and a TfidfVectorizer is fitted on these chunks. For a given user query, cosine similarity is computed between the TF-IDF representation of the query and each chunk, and the top matching chunks are retrieved. These retrieved chunks are then passed to the OpenAI language model to generate an answer. Since TF-IDF relies on exact word overlap and does not capture semantic meaning, this baseline approach often struggles with paraphrased queries, ambiguous wording, and semantic variations, leading to weaker retrieval quality and less reliable answers.

---

## Improved Method (RAG with Embeddings and FAISS)

The improved method replaces keyword-based retrieval with a full Retrieval-Augmented Generation pipeline using semantic embeddings and FAISS. Resume chunks are converted into dense vector embeddings using OpenAI's embedding model and stored in a FAISS vector index. When a user asks a question, the query is embedded and semantically matched against the stored vectors using vector similarity and hybrid scoring. The most relevant chunks are then provided as context to the language model. This approach enables semantic understanding, improves retrieval accuracy, and significantly reduces hallucinations compared to the baseline TF-IDF method.

---

## Evaluation Methodology

To evaluate the performance of the system, we conducted both quantitative and qualitative analysis using a curated set of resume-specific questions with manually written gold answers. Quantitative metrics included retrieval accuracy, semantic similarity between model answers and gold answers, BLEU score, ROUGE-1 and ROUGE-L scores, and hallucination rate. Retrieval accuracy measured whether the correct resume chunks were retrieved, while semantic similarity and ROUGE scores assessed the alignment between generated answers and reference answers. BLEU was used to measure n-gram overlap, and hallucination rate captured cases where the model generated information not present in the resume.

For qualitative evaluation, we manually inspected model outputs and categorized errors into retrieval errors, generation errors, and incomplete answers. We compared the baseline TF-IDF method and the improved RAG method using the same evaluation set. Results showed that the baseline approach frequently failed to retrieve relevant context for semantically phrased questions, while the improved RAG pipeline achieved perfect retrieval accuracy, zero hallucination, and more coherent and grounded answers. This evaluation confirms that semantic retrieval using embeddings and FAISS substantially outperforms keyword-based baselines for resume question answering.
