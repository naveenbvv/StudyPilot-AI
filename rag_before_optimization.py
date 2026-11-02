
import pymupdf
import chromadb
import hashlib
import json
import re

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

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    registry = load_registry()

    # Use all indexed documents if none are specified
    if document_ids is None:
        document_ids = list(registry.keys())

    # Remove duplicate IDs while preserving order
    document_ids = list(dict.fromkeys(document_ids))

    if not document_ids:
        return []

    # Validate selected document IDs
    for document_id in document_ids:

        if document_id not in registry:
            raise ValueError(f"Unknown document ID: {document_id}")

    # Generate question embedding
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    all_passages = []

    # Retrieve passages separately from every selected PDF
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
# 10. EVIDENCE VERIFICATION LAYER
# ==========================================

def verify_answer(answer, context):

    if not answer or not answer.strip():
        raise ValueError("Cannot verify an empty answer.")

    verification_prompt = f"""
You are the evidence editor for StudyPilot AI.

Your task is to EDIT an existing academic answer.
You must not write a completely new answer.

ORIGINAL PDF EVIDENCE:
{context}

ANSWER TO VERIFY:
{answer}

STRICT RULES:

1. Use ONLY the supplied PDF evidence.

2. Preserve every factual claim that is directly supported.

3. Remove unsupported factual claims.
   If important information is missing, write:
   "The provided documents do not establish this."

4. Never replace unsupported information with general knowledge.

5. Check the entire answer for contradictions.
   The introduction, headings, table and conclusion
   must agree with each other.

6. For comparison questions, check each side separately.
   Do not assume that evidence about one concept
   also proves something about the other.

7. Do not invent comparisons about speed, accuracy,
   cost, hardware, performance or applications.

8. Every factual claim must have an appropriate citation
   using the existing format [Source N].

9. Preserve the original source numbers.
   Never invent, renumber or change them.

10. NEVER write filenames or page numbers.
    Python will attach the official source mapping.

11. If a table cell has insufficient evidence,
    write "Not established by the provided documents."

12. Do not create a Sources section.

13. Preserve useful headings, tables and explanations.

14. Make the smallest corrections necessary.

15. Do not add new facts, examples or conclusions.

FINAL CHECK:

Before returning the answer, check that:
- The definition is actually supported.
- Each comparison row is supported.
- Citations refer to supplied source numbers.
- The conclusion agrees with the body.
- No unsupported factual claims remain.

Return ONLY the corrected answer.
"""

    verified_answer = ask_ai(verification_prompt)

    if not verified_answer or not verified_answer.strip():
        raise ValueError(
            "The evidence verification returned an empty answer."
        )

    return verified_answer.strip()


# ==========================================
# 10.1 CITATION VALIDATION
# ==========================================

def clean_citations(answer, source_count):

    if not answer:
        return ""

    # Remove any Sources section written by the AI.
    # Python generates the authoritative one.
    answer = re.sub(
        r"(?im)^\s{0,3}#{1,3}\s*Sources\b.*$",
        "",
        answer
    )

    # Remove source citations containing invented page numbers.
    # Keep only the source number.
    answer = re.sub(
        r"\[\s*Source\s+(\d+)\s*,\s*Page\s+\d+\s*\]",
        r"[Source \1]",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"\(\s*Source\s+(\d+)\s*,\s*Page\s+\d+\s*\)",
        r"[Source \1]",
        answer,
        flags=re.IGNORECASE
    )

    # Validate all remaining standard citations.
    def validate_citation(match):

        source_number = int(match.group(1))

        if 1 <= source_number <= source_count:
            return f"[Source {source_number}]"

        # Invalid source references are removed.
        return ""

    answer = re.sub(
        r"\[\s*Source\s+(\d+)\s*\]",
        validate_citation,
        answer,
        flags=re.IGNORECASE
    )

    # Remove empty parentheses left after citation cleaning.
    answer = re.sub(r"\(\s*\)", "", answer)

    return answer.strip()


# ==========================================
# 10.2 EVIDENCE COVERAGE CHECK
# ==========================================

def check_evidence_coverage(answer, source_count):

    if not answer:
        return answer

    # Detect whether the answer contains any valid source citation.
    valid_citations = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    valid_citations = [
        number for number in valid_citations
        if 1 <= int(number) <= source_count
    ]

    # Do not invent citations.
    # If no citations remain, clearly flag the answer.
    if not valid_citations:
        answer += (
            "\n\n**Evidence warning:** "
            "No valid source citations were found in this answer. "
            "Please check the retrieved PDF passages."
        )

    return answer


# ==========================================
# 11. ASK QUESTIONS USING RAG
# ==========================================

def ask_pdf(question, document_ids=None, top_k=4):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

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

    context_parts = []
    sources = []

    # Build numbered source context
    for i, passage in enumerate(passages, start=1):

        source_name = passage["source"]
        page_number = passage["page"]

        context_parts.append(
            f"""
[Source {i}]
Filename: {source_name}
Page: {page_number}

Content:
{passage["text"]}
"""
        )

        sources.append(
            f"[Source {i}: {source_name}, Page {page_number}]"
        )

    context = "\n".join(context_parts)

    # ======================================
    # INITIAL ANSWER GENERATION
    # ======================================

    prompt = f"""
You are StudyPilot, an academic research assistant.

Answer the student's question using ONLY the supplied PDF evidence.

IMPORTANT RULES:

1. Use only information from DOCUMENT CONTEXT.
2. Never invent facts, examples, statistics or references.
3. Use evidence from all selected documents when relevant.
4. Clearly distinguish information from different documents.
5. Cite factual claims using [Source N].
6. Never invent source numbers.
7. Do not write filenames or page numbers in your answer.
8. Python will generate the official source mapping.
9. If documents disagree, explain the disagreement.
10. If evidence is missing, explicitly say so.
11. Explain technical concepts in beginner-friendly language.
12. Do not use general knowledge to fill gaps in the PDFs.

FOR COMPARISON QUESTIONS:

1. Identify evidence from each document separately.
2. Explain similarities and differences only when supported.
3. Explain relationships between concepts only when supported.
4. Use a Markdown comparison table when useful.
5. Support factual comparison points with citations.
6. Do not claim both documents support something
   unless both actually provide evidence.
7. If a comparison cannot be established,
   write "Not established by the provided documents."

ANSWER FORMAT:

- Start with a short introduction.
- Use clear headings.
- Include comparison tables when appropriate.
- Explain applications only when supported.
- Finish with a concise evidence-based conclusion.
- Do not create a Sources section.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

Write a detailed, well-organized academic answer.
"""

    # Step 1: Generate initial answer
    print("\nGenerating answer...")

    answer = ask_ai(prompt)

    if not answer or not answer.strip():
        raise ValueError("The AI generated an empty answer.")

    # Step 2: Verify answer against original evidence
    print("\nVerifying answer against PDF evidence...")

    answer = verify_answer(answer, context)

    # Step 3: Validate citations using Python
    print("\nValidating citations...")

    answer = clean_citations(
        answer,
        source_count=len(sources)
    )

    # Step 4: Check citation coverage
    answer = check_evidence_coverage(
        answer,
        source_count=len(sources)
    )

    # Step 5: Attach official source mapping
    source_section = "\n".join(sources)

    return f"""{answer}

### Sources

{source_section}
"""


# ==========================================
# 12. TERMINAL TESTING
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
