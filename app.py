
import streamlit as st
import json
import html
from pathlib import Path

from rag import index_pdf, ask_pdf, delete_document


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StudyPilot AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = BASE_DIR / "data" / "documents.json"


# =========================================================
# DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #6258E8;
    --primary-dark: #5046D8;
    --bg: #F7F8FC;
    --surface: #FFFFFF;
    --text: #17192D;
    --muted: #85899B;
    --border: #EAEBF2;
    --sidebar: #17192D;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

/* Main content */

.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid #292B43;
}

section[data-testid="stSidebar"] * {
    color: #F5F5FA;
}

section[data-testid="stSidebar"] .stCaption {
    color: #A7A9BB;
}

section[data-testid="stSidebar"] hr {
    border-color: #34364D;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #22243B;
    border: 1px dashed #565879;
    border-radius: 12px;
    padding: 10px;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #E7E7F0;
}

section[data-testid="stSidebar"] .stButton button {
    background: #6258E8;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: #766CF0;
    border: none;
}

/* Typography */

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text);
    letter-spacing: -0.7px;
}

h1 {
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

/* Buttons */

.stButton button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton button[kind="primary"] {
    background: var(--primary);
    border: 1px solid var(--primary);
    color: white;
}

.stButton button[kind="primary"]:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
}

/* Inputs */

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    border-radius: 10px;
}

.stChatInput {
    border-radius: 16px;
}

/* Cards */

.sp-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 21px;
    min-height: 115px;
    box-shadow: 0 2px 8px rgba(20, 25, 50, 0.025);
}

.sp-card-label {
    color: #85899B;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 9px;
}

.sp-card-value {
    color: #17192D;
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
}

.sp-card-icon {
    font-size: 20px;
    margin-bottom: 12px;
}

.sp-hero {
    background: linear-gradient(115deg, #211E50 0%, #5146C9 65%, #7168EB 100%);
    border-radius: 20px;
    padding: 32px;
    color: white;
    margin-bottom: 25px;
}

.sp-hero h1 {
    color: white !important;
    font-size: 30px;
    margin-bottom: 8px;
}

.sp-hero p {
    color: #E2E0FF;
    font-size: 14px;
    margin-bottom: 0;
}

.sp-section-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 19px;
    color: #17192D;
    margin: 25px 0 13px 0;
}

.sp-document {
    background: white;
    border: 1px solid var(--border);
    padding: 16px;
    border-radius: 13px;
    margin-bottom: 10px;
}

.sp-document-name {
    font-size: 14px;
    font-weight: 700;
    color: #20223A;
    overflow-wrap: anywhere;
}

.sp-document-meta {
    font-size: 12px;
    color: #85899B;
    margin-top: 5px;
}

.sp-tag {
    display: inline-block;
    color: #6258E8;
    background: #F0EFFF;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.sp-empty {
    background: white;
    border: 1px dashed #D9DBE8;
    border-radius: 18px;
    text-align: center;
    padding: 45px 20px;
}

.sp-empty-icon {
    font-size: 38px;
    margin-bottom: 12px;
}

.sp-empty-title {
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #20223A;
}

.sp-empty-text {
    color: #85899B;
    font-size: 13px;
    margin-top: 8px;
}

.sp-footer {
    text-align: center;
    color: #A0A3B1;
    font-size: 12px;
    padding-top: 35px;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 13px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Hide default menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Mobile */

@media(max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 3rem 1rem;
    }

    .sp-hero {
        padding: 23px;
    }

    .sp-hero h1 {
        font-size: 24px;
    }

    .sp-card {
        margin-bottom: 8px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "messages": [],
    "chat_scope": None,
    "app_page": "Dashboard",
    "pending_question": None,
    "delete_confirmation": None
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# DOCUMENT REGISTRY
# =========================================================

def load_documents():
    if not REGISTRY_PATH.exists():
        return []

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        docs = list(data.values()) if isinstance(data, dict) else data
        return sorted(docs, key=lambda d: d.get("name", "").lower())

    except (json.JSONDecodeError, OSError):
        return []


def safe_int(value):
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def document_stats(docs):
    total_pages = sum(safe_int(d.get("pages")) for d in docs)
    total_chunks = sum(safe_int(d.get("chunks")) for d in docs)
    return len(docs), total_pages, total_chunks


# =========================================================
# REUSABLE UI COMPONENTS
# =========================================================

def metric_card(label, value, icon):
    st.markdown(f"""
    <div class="sp-card">
        <div class="sp-card-icon">{icon}</div>
        <div class="sp-card-label">{label}</div>
        <div class="sp-card-value">{value:,}</div>
    </div>
    """, unsafe_allow_html=True)


def document_card(document):
    name = html.escape(str(document.get("name", "Untitled PDF")))
    pages = safe_int(document.get("pages"))
    chunks = safe_int(document.get("chunks"))

    st.markdown(f"""
    <div class="sp-document">
        <div class="sp-document-name">📄 {name}</div>
        <div class="sp-document-meta">
            {pages:,} pages &nbsp;·&nbsp; {chunks:,} indexed chunks
        </div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(title, description):
    st.markdown(f"""
    <div class="sp-empty">
        <div class="sp-empty-icon">📚</div>
        <div class="sp-empty-title">{title}</div>
        <div class="sp-empty-text">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_scope = None
    st.session_state.pending_question = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 18px 0;">
        <div style="font-family:Manrope;font-size:25px;font-weight:800;">
            <span style="color:#9D94FF;">✦</span> StudyPilot
        </div>
        <div style="font-size:12px;color:#A7A9BB;margin-top:4px;">
            YOUR AI ACADEMIC COMPANION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("＋  New Chat", use_container_width=True):
        reset_chat()
        st.session_state.app_page = "Dashboard"
        st.rerun()

    st.markdown("##### WORKSPACE")

    if st.button("⌂  Dashboard", use_container_width=True):
        st.session_state.app_page = "Dashboard"
        st.rerun()

    if st.button("◉  Document Chat", use_container_width=True):
        st.session_state.app_page = "Document Chat"
        st.rerun()

    if st.button("⌕  Research Mode", use_container_width=True):
        st.session_state.app_page = "Research Mode"
        st.rerun()

    if st.button("▤  Document Library", use_container_width=True):
        st.session_state.app_page = "Document Library"
        st.rerun()

    st.divider()

    st.markdown("##### ADD DOCUMENT")

    uploaded_file = st.file_uploader(
        "Upload an academic PDF",
        type=["pdf"],
        help="Choose a PDF file to add to your library."
    )

    if uploaded_file is not None:

        st.caption(f"Selected: {uploaded_file.name}")
        st.caption(f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB")

        if st.button(
            "Process PDF",
            type="primary",
            use_container_width=True
        ):

            pdf_path = UPLOAD_DIR / Path(uploaded_file.name).name

            try:
                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                with st.spinner("Extracting text and indexing your PDF..."):
                    result = index_pdf(pdf_path)

                if isinstance(result, dict):
                    chunk_count = result.get("chunks", 0)
                else:
                    chunk_count = result

                st.success(
                    f"PDF processed successfully. {chunk_count} chunks indexed."
                )

                reset_chat()
                st.rerun()

            except Exception as error:
                st.error(f"Processing failed: {error}")

    st.divider()

    docs = load_documents()

    st.markdown("##### YOUR LIBRARY")

    if docs:

        for doc in docs[:5]:
            name = html.escape(str(doc.get("name", "Untitled")))
            pages = safe_int(doc.get("pages"))

            st.markdown(
                f"""
                <div style="padding:8px 3px;">
                    <div style="font-size:13px;font-weight:600;">
                        📄 {name}
                    </div>
                    <div style="font-size:11px;color:#A7A9BB;">
                        {pages} pages
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if len(docs) > 5:
            st.caption(f"+ {len(docs) - 5} more documents")

    else:
        st.caption("Your library is empty.")

    st.divider()

    st.markdown("""
    <div style="padding:8px 0;">
        <div style="font-size:12px;color:#A7A9BB;">LOCAL AI ENGINE</div>
        <div style="font-size:14px;font-weight:700;margin-top:5px;">
            ● Qwen 2.5 7B
        </div>
        <div style="font-size:11px;color:#A7A9BB;margin-top:3px;">
            Powered by Ollama
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Clear current chat", use_container_width=True):
        reset_chat()
        st.rerun()


# =========================================================
# MAIN PAGE ROUTING
# =========================================================

documents = load_documents()
total_docs, total_pages, total_chunks = document_stats(documents)

page = st.session_state.app_page


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <div class="sp-hero">
        <div style="font-size:12px;font-weight:700;letter-spacing:1.5px;">
            YOUR PERSONAL STUDY WORKSPACE
        </div>
        <h1>Learn smarter. Research deeper.</h1>
        <p>
            Your academic documents, powered by AI.
            Ask questions, understand concepts and discover insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Your workspace")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Documents", total_docs, "📚")

    with c2:
        metric_card("Total pages", total_pages, "📄")

    with c3:
        metric_card("Indexed chunks", total_chunks, "✦")

    st.markdown('<div class="sp-section-title">Get started</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### ◉ Document Chat")
            st.write("Ask questions and understand a specific academic PDF.")
            if st.button("Open Document Chat →", use_container_width=True):
                st.session_state.app_page = "Document Chat"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("### ⌕ Research Mode")
            st.write("Explore and compare information across multiple PDFs.")
            if st.button("Open Research Mode →", use_container_width=True):
                st.session_state.app_page = "Research Mode"
                st.rerun()

    st.markdown('<div class="sp-section-title">Recent documents</div>',
                unsafe_allow_html=True)

    if documents:
        for doc in documents[:4]:
            document_card(doc)

        if st.button("View all documents →"):
            st.session_state.app_page = "Document Library"
            st.rerun()

    else:
        empty_state(
            "Your study library is waiting",
            "Upload your first academic PDF using the sidebar to get started."
        )


# =========================================================
# DOCUMENT LIBRARY
# =========================================================

elif page == "Document Library":

    st.title("Document Library")
    st.caption("Manage all your academic documents in one place.")

    if documents:

        search = st.text_input(
            "Search documents",
            placeholder="Search by document name..."
        )

        filtered_docs = [
            doc for doc in documents
            if search.lower() in str(doc.get("name", "")).lower()
        ]

        st.caption(f"{len(filtered_docs)} document(s) found")

        for doc in filtered_docs:
            document_card(doc)

        st.divider()
        st.subheader("Manage documents")

        doc_options = {
            doc["id"]: doc.get("name", "Untitled PDF")
            for doc in documents
            if doc.get("id")
        }

        if doc_options:

            delete_id = st.selectbox(
                "Choose a document to delete",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                key="library_delete_selection"
            )

            if st.button("Delete selected document", type="secondary"):

                st.session_state.delete_confirmation = delete_id
                st.rerun()

            if st.session_state.delete_confirmation:

                confirm_id = st.session_state.delete_confirmation

                st.warning(
                    f"Are you sure you want to delete "
                    f"{doc_options.get(confirm_id, 'this document')}?"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Confirm deletion", type="primary"):
                        try:
                            delete_document(confirm_id)
                            st.session_state.delete_confirmation = None
                            reset_chat()
                            st.success("Document deleted.")
                            st.rerun()
                        except Exception as error:
                            st.error(f"Deletion failed: {error}")

                with col2:
                    if st.button("Cancel"):
                        st.session_state.delete_confirmation = None
                        st.rerun()

    else:
        empty_state(
            "No documents yet",
            "Upload a PDF from the sidebar to build your study library."
        )


# =========================================================
# DOCUMENT CHAT / RESEARCH MODE
# =========================================================

elif page in ["Document Chat", "Research Mode"]:

    is_research = page == "Research Mode"

    if is_research:
        st.title("Research Mode")
        st.caption(
            "Explore information and discover connections across multiple PDFs."
        )
    else:
        st.title("Document Chat")
        st.caption(
            "Have a focused conversation with your academic document."
        )

    if not documents:

        empty_state(
            "Upload a document to begin",
            "Your AI study workspace will appear here once you process a PDF."
        )

    else:

        document_options = {
            doc["id"]: doc["name"]
            for doc in documents
            if doc.get("id")
        }

        if is_research:

            selected_ids = st.multiselect(
                "Choose documents for research",
                options=list(document_options.keys()),
                default=list(document_options.keys()),
                format_func=lambda doc_id: document_options[doc_id],
                key="research_document_selection"
            )

            st.markdown(
                f'<span class="sp-tag">{len(selected_ids)} DOCUMENT(S) SELECTED</span>',
                unsafe_allow_html=True
            )

        else:

            selected_id = st.selectbox(
                "Choose your study document",
                options=list(document_options.keys()),
                format_func=lambda doc_id: document_options[doc_id],
                key="document_chat_selection"
            )

            selected_ids = [selected_id]

            selected_doc = next(
                (d for d in documents if d.get("id") == selected_id),
                {}
            )

            st.markdown(
                f"""
                <div class="sp-document">
                    <div class="sp-document-name">
                        📄 {html.escape(str(selected_doc.get('name', 'Document')))}
                    </div>
                    <div class="sp-document-meta">
                        {safe_int(selected_doc.get('pages'))} pages
                        &nbsp;·&nbsp;
                        {safe_int(selected_doc.get('chunks'))} indexed chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        chat_scope = (page, tuple(selected_ids))

        if st.session_state.chat_scope != chat_scope:
            st.session_state.messages = []
            st.session_state.chat_scope = chat_scope

        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown("#### Conversation")

        with col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if not st.session_state.messages:

            st.markdown("##### What would you like to explore?")

            suggestions = [
                "Summarize the main topics",
                "Explain the important concepts",
                "What are the key takeaways?",
            ]

            suggestion_columns = st.columns(3)

            for i, suggestion in enumerate(suggestions):
                with suggestion_columns[i]:
                    if st.button(suggestion, use_container_width=True):
                        st.session_state.pending_question = suggestion
                        st.rerun()

        # Render conversation

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Pending suggested question

        question = st.session_state.pending_question

        if question:
            st.session_state.pending_question = None
        else:
            question = st.chat_input(
                "Ask anything about your study materials..."
            )

        # Process question

        if question:

            if not selected_ids:

                st.warning("Select at least one document first.")

            else:

                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):

                    with st.spinner("StudyPilot is thinking..."):

                        try:

                            answer = ask_pdf(
                                question,
                                document_ids=selected_ids
                            )

                            st.markdown(answer)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })

                        except Exception as error:

                            error_message = (
                                f"Something went wrong: {error}"
                            )

                            st.error(error_message)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="sp-footer">
    STUDYPILOT AI &nbsp;·&nbsp; YOUR PERSONAL AI ACADEMIC COMPANION
    <br><br>
    Locally powered by Ollama and your academic documents.
</div>
""", unsafe_allow_html=True)
