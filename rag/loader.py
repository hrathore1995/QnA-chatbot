import pdfplumber
from docx import Document

# cleaning text
def clean_text(text):
    # removing none, trimming, normalizing spaces
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")
    text = text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip() != ""]
    return "\n".join(lines)

# extracting text from pdf
def load_pdf(file):
    full_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            cleaned = clean_text(text)
            if cleaned:
                full_text.append(cleaned)
    return "\n".join(full_text)

# extracting text from docx
def load_docx(file):
    doc = Document(file)
    lines = [para.text for para in doc.paragraphs]
    cleaned_lines = [clean_text(line) for line in lines if clean_text(line)]
    return "\n".join(cleaned_lines)

# loading file based on type
def load_resume(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        return load_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        return load_docx(uploaded_file)
    else:
        return ""
