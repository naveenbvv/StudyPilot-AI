
import pymupdf
import chromadb
import hashlib
import json

from pathlib import Path
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from ollama_client import ask_ai


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "chroma_db"
REGISTRY_FILE = DATA_DIR / "documents.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "studypilot_documents"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 3. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 4. DOCUMENT REGISTRY
# ==========================================

def load_registry():

    if not REGISTRY_FILE.exists():
        return {}

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry):

    with open(REGISTRY_FILE, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


def list_documents():

    registry = load_registry()
    return list(registry.values())


def get_document(document_id):

    registry = load_registry()
    return registry.get(document_id)


# ==========================================
# 5. GENERATE UNIQUE DOCUMENT ID
# ==========================================

def generate_document_id(pdf_path):

    with open(pdf_path, "rb") as file:
        file_hash = hashlib.sha256(file.read()).hexdigest()

    return file_hash


# ==========================================
# 6. SPLIT PDF TEXT INTO CHUNKS
# ==========================================

def split_text(text, chunk_size=1000, overlap=150):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap value.")

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ==========================================
# 7. INDEX A PDF
# ==========================================

def index_pdf(pdf_path):

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    document_id = generate_document_id(pdf_path)

    registry = load_registry()

    # Prevent indexing identical PDF content twice
    if document_id in registry:

        existing = registry[document_id]

        print(f"Already indexed: {existing['name']}")

        return {
            "document_id": document_id,
            "chunks": existing["chunks"],
            "already_indexed": True
        }

    print(f"Reading PDF: {pdf_path.name}")

    document = pymupdf.open(str(pdf_path))

    documents = []
    metadatas = []
    ids = []

    try:

        page_count = len(document)

        for page_number, page in enumerate(document):

            text = page.get_text("text")

            chunks = split_text(text)

            for chunk_number, chunk in enumerate(chunks):

                chunk_id = hashlib.sha256(
                    f"{document_id}:{page_number}:{chunk_number}".encode()
                ).hexdigest()

                documents.append(chunk)

                metadatas.append({
                    "document_id": document_id,
                    "source": pdf_path.name,
                    "page": page_number + 1,
                    "chunk": chunk_number
                })

                ids.append(chunk_id)

    finally:
        document.close()

    if not documents:
        raise ValueError(
            "No extractable text found. "
            "This PDF may be scanned or image-only."
        )

    print(f"Creating embeddings for {len(documents)} chunks...")

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    registry[document_id] = {
        "id": document_id,
        "name": pdf_path.name,
        "chunks": len(documents),
        "pages": page_count,
        "indexed_at": datetime.now(timezone.utc).isoformat()
    }

    save_registry(registry)

    print("PDF successfully indexed!")

    return {
        "document_id": document_id,
        "chunks": len(documents),
        "already_indexed": False
    }


# ==========================================
# 8. DELETE A DOCUMENT
# ==========================================

def delete_document(document_id):

    registry = load_registry()

    if document_id not in registry:
        raise ValueError("Document not found.")

    collection.delete(
        where={"document_id": document_id}
    )

    del registry[document_id]

    save_registry(registry)

    return True


# ==========================================
# 9. RETRIEVE RELEVANT DOCUMENT CONTENT
# ==========================================

def retrieve_context(question, document_ids=None, top_k=4):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Load the document registry
    registry = load_registry()

    # Use all documents if none are specified
    if document_ids is None:
        document_ids = list(registry.keys())

    if not document_ids:
        return []

    # Validate selected documents
    for document_id in document_ids:
        if document_id not in registry:
            raise ValueError(f"Unknown document ID: {document_id}")

    # Generate question embedding
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    all_passages = []

    # Search each selected PDF separately
    for document_id in document_ids:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text_content, metadata, distance in zip(
            documents, metadatas, distances
        ):
            all_passages.append({
                "text": text_content,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance
            })

    # Sort by relevance
    all_passages.sort(key=lambda item: item["distance"])

    # Remove internal distance values
    for passage in all_passages:
        passage.pop("distance")

    return all_passages

# ==========================================
# 10. ASK QUESTIONS USING RAG
# ==========================================

def ask_pdf(question, document_ids=None, top_k=4):

    passages = retrieve_context(
        question,
        document_ids=document_ids,
        top_k=top_k
    )

    if not passages:
        return (
            "I couldn't find relevant information "
            "in the selected documents."
        )

    context = ""
    sources = []

    for i, passage in enumerate(passages, start=1):

        source_name = passage["source"]
        page_number = passage["page"]

        context += f"""
[Source {i}]
Filename: {source_name}
Page: {page_number}

Content:
{passage["text"]}
"""

        sources.append(
            f"[Source {i}: {source_name}, Page {page_number}]"
        )

    prompt = f"""
You are StudyPilot, an academic research assistant.

Answer the student's question using ONLY the supplied document context.

Rules:
1. Do not invent facts.
2. If the context is insufficient, say so clearly.
3. Explain concepts in beginner-friendly language.
4. Cite important claims using the exact source numbers provided.
5. Use citations like [Source 1] or [Source 2].
6. Never invent source numbers, filenames, or page numbers.
7. Distinguish information from different documents.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

Write a clear, structured answer with source citations.
"""

    answer = ask_ai(prompt)

    source_section = "\n".join(sources)

    return f"""{answer}

### Sources
{source_section}
"""


# ==========================================
# 11. TERMINAL TESTING
# ==========================================

if __name__ == "__main__":

    print("\n===== STUDYPILOT DOCUMENT INTELLIGENCE =====")

    while True:

        print("\n1. Index a PDF")
        print("2. List documents")
        print("3. Delete a document")
        print("4. Ask a question")
        print("5. Exit")

        choice = input("\nChoose an option: ").strip()

        try:

            if choice == "1":

                path = input(
                    "Enter PDF path: "
                ).strip().strip('"')

                result = index_pdf(path)

                print(result)

            elif choice == "2":

                for doc in list_documents():
                    print(doc)

            elif choice == "3":

                for doc in list_documents():
                    print(doc["id"], "-", doc["name"])

                document_id = input(
                    "Enter document ID to delete: "
                ).strip()

                delete_document(document_id)

                print("Document deleted successfully.")

            elif choice == "4":

                for doc in list_documents():
                    print(doc["id"], "-", doc["name"])

                ids = input(
                    "Enter document IDs separated by commas (or ALL): "
                ).strip()

                if ids.upper() == "ALL":

                    selected_ids = None

                else:

                    selected_ids = [
                        item.strip()
                        for item in ids.split(",")
                        if item.strip()
                    ]

                question = input("Ask your question: ")

                answer = ask_pdf(
                    question,
                    document_ids=selected_ids
                )

                print("\nStudyPilot:\n")
                print(answer)

            elif choice == "5":

                print("Exiting StudyPilot.")
                break

            else:

                print("Invalid option.")

        except Exception as error:

            print(f"Error: {error}")
