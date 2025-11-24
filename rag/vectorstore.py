import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# loading api key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# creating embeddings for text list
def embed_text_list(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors).astype("float32")

# building faiss index from chunks
def build_faiss_index(chunks):
    vectors = embed_text_list(chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, vectors

# retrieving top k similar chunks
def search_faiss(query, chunks, index, k=5):
    # getting initial vector results
    query_vector = embed_text_list([query])
    distances, indices = index.search(query_vector, k)

    results = []
    for rank, i in enumerate(indices[0]):
        if i < len(chunks):
            chunk = chunks[i]

            # computing simple keyword overlap score
            query_words = set(query.lower().split())
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))

            # computing hybrid score
            vector_score = 1 / (1 + distances[0][rank])
            hybrid_score = vector_score + (0.1 * overlap)

            results.append((chunk, hybrid_score))

    # sorting by score
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # returning only chunks
    return [chunk for chunk, score in results]

