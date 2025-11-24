# splitting text into chunks
def split_text(text, max_length=800, overlap=200):
    chunks = []
    start = 0
    end = max_length

    # creating chunks with overlap
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + max_length

    return chunks
