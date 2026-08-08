"""
ingest.py — Knowledge Base Builder for virtual-twin
=====================================================
Run this ONCE before starting the app (or whenever you update the resume):

    python ingest.py

What it does:
  1. Loads data/resume.pdf with pypdf
  2. Loads data/interview_context.txt (deep profile)
  3. Loads inline EXTRA_KNOWLEDGE FAQ block
  4. Splits into overlapping chunks (optimised for RAG retrieval)
  5. Embeds in batches of 80 (Gemini free tier: 100 req/min limit)
  6. Persists the vector store to ./chroma_db

After running, the RAG agent in graph.py will use this store automatically.

Free-tier note: ~161 chunks take ~2 minutes to embed (rate-limit sleep between batches).
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # loads .env locally; on Streamlit Cloud use Secrets

RESUME_PATH = os.path.join("data", "resume.pdf")
INTERVIEW_CONTEXT_PATH = os.path.join("data", "interview_context.txt")
CHROMA_DIR = "./chroma_db"
COLLECTION = "sowbhagya_profile"


# ── Additional plain-text knowledge you want in the KB ───────────────────────
# Any key Q&A, bio text, or facts that are NOT in the PDF can go here.

EXTRA_KNOWLEDGE = """
Name: Sowbhagya Mohanthy
Role: AI Engineer & Software Developer
Company: IBM — CIO Agile & Emerging Technologies
Location: Hyderabad, India
LinkedIn: https://linkedin.com/in/sowbhagya-mohanthy
HuggingFace: https://huggingface.co/Sowbhagya-45
Email: sudhirkumar02001@gmail.com
Phone: +91 93467 05286 (share only if visitor explicitly asks)
Experience: 3 years at IBM (July 2023 – Present, including internship from Jan 2023)

Current Availability:
Sowbhagya is open to AI engineering opportunities, consulting conversations, and
interesting collaboration projects. He is currently employed full-time at IBM.

Key expertise areas:
- Agentic AI systems: LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, MCP
- LLM Engineering: IBM Granite, LangChain, RAG, Prompt Engineering
- AI/ML: UMAP, HDBSCAN, ensemble classifiers, semantic embeddings
- Backend: Python 3.12, FastAPI, APScheduler, Pydantic v2, Node.js
- Frontend: React.js, Next.js, Redux
- Databases: MongoDB, PostgreSQL, Milvus DB, Redis, FAISS
- DevOps: IBM Cirrus, OpenShift, Docker, CI/CD

Flagship project:
Recommendation Engine AI (Production) at IBM — a 5-stage automated AI pipeline
that converts raw enterprise customer feedback into business recommendations,
deployed on IBM Cirrus/OpenShift. Tech: Python 3.12, FastAPI, IBM Granite,
UMAP+HDBSCAN semantic clustering, 3-tier ensemble classifier.

Education:
B.Tech in Computer Science & Engineering | GPA 9.2/10
Gokaraju Rangaraju Institute of Engineering & Technology, 2019–2023

Certifications:
- IBM WatsonX Essentials (2024)
- IBM Agile Explorer
- Enterprise Design Thinking Practitioner — IBM
- IBM Developer Jumpstart Practitioner
- CCNA: Introduction to Networks — Cisco (2022)
- AWS Academy Cloud Foundations — Amazon Web Services (2021)

Frequently asked questions:
Q: What is Sowbhagya's notice period?
A: He is currently employed; notice period would depend on the role and company.

Q: Is Sowbhagya open to remote roles?
A: Yes, he is open to remote, hybrid, and on-site roles across India.

Q: What kind of roles is Sowbhagya looking for?
A: AI Engineer, ML Engineer, LLM/Agentic AI specialist, Full-Stack AI Developer roles.

Q: Does Sowbhagya know LangGraph?
A: Yes — LangGraph is one of his primary tools. He has hands-on production experience
   building multi-agent systems, supervisor patterns, and stateful RAG pipelines with it.
"""


def ingest() -> None:
    """Main ingestion function."""
    # Validate environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set. Add it to .env or environment.")

    if not os.path.exists(RESUME_PATH):
        sys.exit(f"ERROR: Resume not found at {RESUME_PATH}")

    print("virtual-twin | Knowledge Base Ingestion")
    print("=" * 45)

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    print(f"[1/4] Loading resume from {RESUME_PATH} ...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_community.document_loaders import PyPDFLoader

    pdf_docs = PyPDFLoader(RESUME_PATH).load()
    print(f"      Loaded {len(pdf_docs)} page(s) from PDF.")

    # ── Step 2: Load interview_context.txt + inline FAQ ───────────────────────
    from langchain_core.documents import Document

    context_docs = []
    if os.path.exists(INTERVIEW_CONTEXT_PATH):
        with open(INTERVIEW_CONTEXT_PATH, "r", encoding="utf-8") as f:
            context_text = f.read()
        context_docs = [Document(
            page_content=context_text,
            metadata={"source": "interview_context", "type": "deep_profile"},
        )]
        print(f"      Loaded interview_context.txt ({len(context_text):,} chars).")
    else:
        print(f"      interview_context.txt not found — skipping.")

    extra_doc = Document(
        page_content=EXTRA_KNOWLEDGE,
        metadata={"source": "extra_knowledge", "type": "faq_and_bio"},
    )
    all_docs = pdf_docs + context_docs + [extra_doc]

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    print("[2/4] Splitting documents into chunks ...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    # smaller = fewer tokens per retrieval call
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"      Created {len(chunks)} chunks.")

    # ── Step 4: Embed + store ─────────────────────────────────────────────────
    print("[3/4] Embedding with Gemini gemini-embedding-001 ...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # Wipe and recreate collection for clean rebuilds
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print(f"      Cleared existing collection '{COLLECTION}'.")
    except Exception:
        pass

    # ── Step 4: Embed one-by-one with throttling ──────────────────────────────
    # Gemini free tier: 100 embed requests/minute.
    # We embed each chunk individually with a 0.7s sleep → ~85 req/min, safe.
    # This bypasses LangChain's internal batching which doesn't throttle.
    REQ_DELAY = 0.7  # seconds between individual embed calls

    print(f"[4/4] Embedding {len(chunks)} chunks one-by-one ({REQ_DELAY}s delay) ...")
    print(f"      Estimated time: ~{int(len(chunks) * REQ_DELAY / 60) + 1} min")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    all_embeddings = []
    for idx, text in enumerate(texts):
        if idx > 0 and idx % 10 == 0:
            print(f"      [{idx}/{len(texts)}] embedded...", flush=True)
        vec = embeddings.embed_query(text)  # single-doc call, no internal batching
        all_embeddings.append(vec)
        time.sleep(REQ_DELAY)

    print(f"      [{len(texts)}/{len(texts)}] all chunks embedded. Persisting ...")

    # Build the collection directly from pre-computed embeddings (no API calls)
    import chromadb as _chromadb
    from langchain_chroma import Chroma as _Chroma
    import uuid as _uuid

    raw_client = _chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        raw_client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = raw_client.create_collection(COLLECTION)
    coll.add(
        ids=[str(_uuid.uuid4()) for _ in chunks],
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    # Wrap in LangChain Chroma for smoke test
    vectorstore = _Chroma(
        client=raw_client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    print(f"\n✅ Ingestion complete! {len(chunks)} chunks stored in '{CHROMA_DIR}'.")
    print("   Run `streamlit run app.py` to start virtual-twin.\n")

    # Quick smoke test
    results = vectorstore.similarity_search("LangGraph experience", k=2)
    print("Smoke test — top 2 results for 'LangGraph experience':")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.page_content[:120].strip()}...")


if __name__ == "__main__":
    ingest()
