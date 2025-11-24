import streamlit as st

from rag.loader import load_resume
from rag.rag_pipeline import build_resume_kb, answer_query

# injecting custom css
def load_css():
    css = """
    <style>
        .user-msg {
            background-color: #DCF2FF;
            padding: 10px;
            border-radius: 12px;
            margin-bottom: 8px;
            color: #003A5D;
        }
        .assistant-msg {
            background-color: #E8E8E8;
            padding: 10px;
            border-radius: 12px;
            margin-bottom: 8px;
            color: #1A1A1A;
        }
        .stChatMessage div {
            padding: 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# setting page config
st.set_page_config(page_title="Resume QnA Chatbot", layout="wide")

# initializing session state
if "kb" not in st.session_state:
    st.session_state.kb = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# creating title
st.title("Resume QnA Chatbot")

# loading resume
uploaded_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])

# handling resume upload
if uploaded_file is not None and st.session_state.kb is None:
    # loading text
    resume_text = load_resume(uploaded_file)
    st.session_state.resume_text = resume_text

    # validating extracted text
    if len(resume_text.strip()) < 100:
        st.write("Resume text is too short or could not be extracted.")
    else:
        # building knowledge base
        st.session_state.kb = build_resume_kb(resume_text)
        st.write("Resume processed. You can now ask questions.")
        
        # showing stats
        st.write(f"Characters extracted: {len(resume_text)}")
        st.write(f"Approx chunks: {len(st.session_state.kb['chunks'])}")

        # showing preview
        with st.expander("Show extracted resume text"):
            st.text(resume_text)

# showing chat history with colored bubbles
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# getting user query
user_query = st.chat_input("Ask a question about the resume")

# handling user query
if user_query:
    if st.session_state.kb is None:
        with st.chat_message("assistant"):
            st.markdown("<div class='assistant-msg'>Upload a resume first.</div>", unsafe_allow_html=True)
    else:
        # adding user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-msg'>{user_query}</div>", unsafe_allow_html=True)

        # generating answer with spinner
        with st.spinner("Processing answer..."):
            answer = answer_query(user_query, st.session_state.kb)

        # adding assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='assistant-msg'>{answer}</div>", unsafe_allow_html=True)

# handling clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.kb = None
    st.session_state.resume_text = ""
    st.rerun()
