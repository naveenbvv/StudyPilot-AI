
import streamlit as st
from pathlib import Path

from rag import index_pdf, ask_pdf

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="StudyPilot AI",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Session state
# -----------------------------

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("StudyPilot AI")
    st.caption("Your personal AI study assistant")

    st.divider()

    st.subheader("Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("Process PDF", use_container_width=True):

            pdf_path = UPLOAD_DIR / uploaded_file.name

            try:
                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                with st.spinner("Reading and indexing your PDF..."):
                    chunk_count = index_pdf(pdf_path)

                st.session_state.indexed = True
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully! {chunk_count} chunks indexed."
                )

            except Exception as error:
                st.error(f"Processing failed: {error}")

    st.divider()

    if st.session_state.indexed:
        st.success("Document ready")
        st.caption(f"File: {st.session_state.document_name}")
    else:
        st.info("Upload and process a PDF to begin.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Main interface
# -----------------------------

st.title("StudyPilot AI")
st.subheader("Understand your study materials with AI.")

st.write(
    "Upload your academic PDF and ask questions. "
    "StudyPilot will retrieve relevant information and generate answers."
)

st.divider()

if not st.session_state.indexed:

    st.info("Upload a PDF from the sidebar and click Process PDF.")

else:

    st.markdown(f"**Currently loaded:** {st.session_state.document_name}")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Question input
    question = st.chat_input("Ask something about your PDF...")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("StudyPilot is thinking..."):

                try:
                    answer = ask_pdf(question)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(f"Something went wrong: {error}")
