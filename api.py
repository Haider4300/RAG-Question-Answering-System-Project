from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime
import asyncio
import logging
import threading
import os
import uuid
import json
import time

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
STORAGE_DIR          = "uploaded_files"
CHAT_HISTORY_FILE    = "chat_history.json"
DOCUMENTS_META_FILE  = "documents_meta.json"   # NEW: persist doc metadata
INDEX_DIR            = "faiss_index_uploaded"
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 200
TOP_K                = 3        # default chunks for specific questions
TOP_K_BROAD          = 6        # more chunks for broad/summary questions
MAX_CONTEXT_CHARS    = 3000     # enough context for meaningful doc summaries
EMBEDDING_BATCH_SIZE = 20
MAX_FILE_SIZE        = 10 * 1024 * 1024   # 10 MB
MAX_PAGES            = 250
MAX_CHUNKS           = 500

# Cosine relevance threshold — 0 to 1, higher = stricter match required
# 0.30 is permissive enough for general doc questions like "tell me about this doc"
RELEVANCE_THRESHOLD  = 0.40   # strict enough to avoid garbage chunks

# ─── Globals ──────────────────────────────────────────────────────────────────
_llm         = None
_embeddings  = None
_vectorstore = None
_retriever   = None
_web_search  = None

# Thread-safe processing lock
_processing_lock = threading.Lock()

# In-memory DBs (persisted to disk on shutdown)
documents_db: dict = {}
chats_db:     dict = {}

# Per-chat active document: remembers which doc the user is focused on
# so follow-up questions ("what about the skills?") use the correct doc
_active_doc: dict = {}   # { chat_id: doc_id }

# Per-document indexing status
doc_status: dict = {}

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,   exist_ok=True)


# ─── Lazy singletons ──────────────────────────────────────────────────────────

def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model="gemma3:4b",   # Better instruction following than llama3.2:1b
            temperature=0,
            num_predict=512,
            num_ctx=4096,        # gemma3 handles longer context well
        )
    return _llm


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return _embeddings


def get_web_search() -> DuckDuckGoSearchRun:
    global _web_search
    if _web_search is None:
        _web_search = DuckDuckGoSearchRun()
    return _web_search


# ─── Query classifier ─────────────────────────────────────────────────────────


def classify_query(question: str) -> str:
    """
    Rule-based classifier — returns DOCUMENT | GENERAL | WEB.
    Priority order: Greeting → Doc signals → WEB signals → Knowledge → Action → Default
    """
    q = question.strip().lower()

    # ── 1. Greetings & small talk → GENERAL ───────────────────────────────────
    GREETING_EXACT = {
        "hi", "hey", "hello", "hiya", "howdy", "greetings",
        "sup", "yo", "heya", "hai", "bye", "goodbye", "thanks",
        "thank you", "ok", "okay", "sure", "great", "nice",
    }
    GREETING_STARTERS = (
        "hi ", "hi!", "hi,",
        "hey ", "hey!", "hey,",
        "hello ", "hello!", "hello,",
        "hiya", "howdy", "good morning", "good afternoon",
        "good evening", "good night", "what's up", "whats up",
        "how are you", "how r u", "how do you do",
    )
    SMALL_TALK_EXACT = {
        "what's going on", "whats going on", "what is going on",
        "what's happening", "whats happening",
        "what's new", "whats new",
        "how's it going", "hows it going",
        "what can you do", "what do you do",
        "who are you", "what are you", "who made you",
        "what's your name", "whats your name", "what is your name",
        "are you an ai", "are you a bot", "are you real",
        "tell me about yourself", "introduce yourself",
        "how do you work",
    }
    if q in GREETING_EXACT:
        return "GENERAL"
    if any(q.startswith(g) for g in GREETING_STARTERS):
        return "GENERAL"
    if q in SMALL_TALK_EXACT or any(q.startswith(t) for t in SMALL_TALK_EXACT):
        return "GENERAL"
    if len(q) <= 10 and not any(w in q for w in ["doc", "pdf", "file", "web", "news", "cv"]):
        return "GENERAL"

    # ── 2. Document signals → DOCUMENT (only if docs uploaded) ────────────────
    if len(documents_db) > 0:
        VAGUE_DOC = [
            "this doc", "my doc", "the doc",
            "this pdf", "my pdf", "the pdf",
            "this file", "my file", "the file",
            "this paper", "my paper", "the paper",
            "this report", "my report", "the report",
            "this document", "my document", "the document",
            "this assignment", "my assignment", "the assignment",
            "this text", "this article",
            "this cv", "my cv", "the cv", "this resume", "my resume",
            "in it", "about it", "from it",
            "summarize the", "summary of the",
            "from the document", "in the document",
            "from my pdf", "in my pdf",
            "tell me about this",
        ]
        if any(t in q for t in VAGUE_DOC):
            return "DOCUMENT"

        # Explicit filename match
        for data in documents_db.values():
            fname      = data.get("filename", "")
            fname_stem = fname.rsplit(".", 1)[0].lower()
            if (fname_stem and len(fname_stem) > 3 and fname_stem in q) \
               or fname.lower() in q:
                return "DOCUMENT"

        # CV/resume domain words → DOCUMENT when files are uploaded
        CV_WORDS = [
            "certification", "certifications", "certificate",
            "skills", "experience", "education", "qualification",
            "work history", "employment", "projects", "achievements",
            "objective", "summary", "profile", "contact",
        ]
        # Only route to DOCUMENT for cv words if a CV-like file is uploaded
        cv_uploaded = any(
            any(kw in data.get("filename", "").lower()
                for kw in ["cv", "resume", "curriculum"])
            for data in documents_db.values()
        )
        if cv_uploaded and any(w in q for w in CV_WORDS):
            return "DOCUMENT"

    # ── 3. Web / live-data signals → WEB ──────────────────────────────────────
    WEB_TRIGGERS = [
        # Weather — all variants
        "current weather", "weather today", "weather of ", "weather in ",
        "temperature in ", "temperature of ", "temperature today",
        "today's temperature", "today's weather", "today's forecast",
        "forecast for ", "climate in ",
        # News
        "latest news", "breaking news", "news today", "today's news",
        "current events", "what happened today", "recently announced",
        # Prices / finance
        "current price", "live score", "stock price",
        "price of bitcoin", "price of gold", "exchange rate",
        "petrol price", "fuel price", "oil price",
        "who won the match", "match result",
        # Explicit search requests
        "do a search", "search for", "look up", "find out the current",
        # Current leadership
        "current president", "current prime minister", "current pm",
        "current cm", "current chief minister", "current governor",
        "current ceo", "current chairman", "current minister",
        "current leader", "current head",
        # Current rankings/lists
        "current top ", "current list", "current ranking",
        "top 5 ", "top 10 ", "top 3 ",   # ranked lists of real-world entities
        # Economic indicators
        "current gdp", "gdp of ", "current inflation", "inflation rate of ",
        "current unemployment", "economic growth of ",
        # Factual lists
        "universities in ", "colleges in ",
        "list of universities", "list of colleges",
        # Population / demographics
        "population of ", "population in ",
    ]
    if any(t in q for t in WEB_TRIGGERS):
        return "WEB"

    # ── 4. Knowledge / conceptual questions → GENERAL ─────────────────────────
    import re as _re

    # Detect year references (1900–2099)
    has_year = bool(_re.search(r'\b(19|20)\d{2}\b', q))

    # Real-world factual topics that need current data, not LLM training knowledge
    FACTUAL_CONTEXT = [
        # Countries & regions
        "pakistan", "india", "china", "usa", "uk", "iran", "turkey",
        "france", "germany", "italy", "russia", "brazil", "canada",
        "bangladesh", "afghanistan", "saudi arabia", "uae", "dubai",
        "lahore", "karachi", "islamabad", "delhi", "beijing",
        # Factual nouns requiring live/current lookup
        "population", "gdp", "inflation", "unemployment", "growth rate",
        "president", "prime minister", "chief minister", "governor",
        "election", "parliament", "government", "policy",
        "price", "rate", "index", "ranking", "score",
        # Military / geopolitical
        "nuclear", "atomic", "warhead", "missile", "military",
        "army", "navy", "airforce", "troops", "defense",
        "countries with", "nations with", "states with",
    ]
    has_factual_context = any(t in q for t in FACTUAL_CONTEXT)

    KNOWLEDGE_STARTERS = (
        "what is ", "what was ", "what were ", "what are ",
        "how does ", "how do ", "how is ", "how are ",
        "why is ", "why are ", "why does ", "why do ",
        "explain ", "define ", "describe ",
        "can you explain", "can you describe",
        "difference between", "compare ",
        "what's the difference", "pros and cons",
        "who is ", "who was ", "who are ",
    )
    if any(q.startswith(s) for s in KNOWLEDGE_STARTERS):
        # If the query is about a real-world fact with a year or factual noun → WEB
        if has_year or has_factual_context:
            # But only if no doc context — doc questions like "what is this paper about" must stay DOCUMENT
            if len(documents_db) > 0:
                DOC_CONTEXT = [
                    "the paper", "this paper", "the doc", "this doc",
                    "the pdf", "this pdf", "the report", "the document",
                    "the assignment", "the article", "the study",
                    "the cv", "this cv", "the resume",
                    "uploaded", "you have", "i gave",
                    "in it", "about it",
                ]
                if any(t in q for t in DOC_CONTEXT):
                    return "DOCUMENT"
            return "WEB"

        # Pure conceptual/knowledge question → check for doc context first
        if len(documents_db) > 0:
            DOC_CONTEXT = [
                "the paper", "this paper", "the doc", "this doc",
                "the pdf", "this pdf", "the report", "the document",
                "the assignment", "the article", "the study",
                "the cv", "this cv", "the resume",
                "uploaded", "you have", "i gave",
            ]
            if any(t in q for t in DOC_CONTEXT):
                return "DOCUMENT"
        return "GENERAL"

    # ── 5. Action words implying operate on uploaded doc → DOCUMENT ───────────
    if len(documents_db) > 0:
        ACTION_ON_DOC = [
            "summarize", "summary", "overview", "main points",
            "key points", "features", "highlights", "conclusion",
            "introduction", "abstract", "findings", "results",
            "methodology", "tell me more", "more details", "in short",
            "briefly", "in brief", "tldr", "tl;dr",
            # CV-specific sections
            "skills", "experience", "certifications", "education",
            "projects", "achievements", "work history",
        ]
        if any(t in q for t in ACTION_ON_DOC):
            return "DOCUMENT"

    # ── 6. Default → GENERAL ──────────────────────────────────────────────────
    return "GENERAL"


# ─── Document processing helpers ──────────────────────────────────────────────

def process_pdf(path: str) -> str:
    pages = []
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        for i, page in enumerate(loader.lazy_load()):
            if i >= MAX_PAGES:
                break
            if page.page_content.strip():
                pages.append((i + 1, page.page_content))
    except Exception as e:
        logger.warning(f"PyPDFLoader failed: {e}")

    if not pages:
        try:
            import pymupdf
            doc = pymupdf.open(path)
            for i in range(min(len(doc), MAX_PAGES)):
                text = doc[i].get_text()
                if text.strip():
                    pages.append((i + 1, text))
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF fallback failed: {e}")

    return "\n\n".join(f"[Page {n}]\n{t}" for n, t in pages)


def process_docx(path: str) -> str:
    from docx import Document as DocxDocument
    return "\n".join(p.text for p in DocxDocument(path).paragraphs if p.text.strip())


def process_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_page_number(text: str) -> int:
    import re
    m = re.search(r"\[Page (\d+)\]", text)
    return int(m.group(1)) if m else 0


def strip_page_markers(text: str) -> str:
    import re
    return re.sub(r"\[Page \d+\]\n?", "", text).strip()


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ─── Persistence helpers ───────────────────────────────────────────────────────

def save_documents_meta():
    """Persist documents_db (without large text_content) to disk."""
    slim = {}
    for doc_id, data in documents_db.items():
        slim[doc_id] = {
            "id":          data["id"],
            "filename":    data["filename"],
            "file_type":   data["file_type"],
            "uploaded_at": data["uploaded_at"],
            "file_path":   data["file_path"],
            # store text length, not content — saves disk space
            "text_length": len(data.get("text_content", "")),
        }
    with open(DOCUMENTS_META_FILE, "w") as f:
        json.dump(slim, f, indent=2)


def load_documents_meta():
    global documents_db
    if not os.path.exists(DOCUMENTS_META_FILE):
        return
    try:
        with open(DOCUMENTS_META_FILE) as f:
            data = json.load(f)
        for doc_id, meta in data.items():
            # Only restore if file still exists on disk
            if os.path.exists(meta.get("file_path", "")):
                documents_db[doc_id] = {**meta, "text_content": ""}
                doc_status[doc_id] = "ready"   # assume indexed from saved FAISS
    except Exception as e:
        logger.warning(f"Could not load documents meta: {e}")


def save_chats():
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump({"chats": list(chats_db.values())}, f, indent=2, default=str)


def load_chats():
    global chats_db
    if not os.path.exists(CHAT_HISTORY_FILE):
        return
    try:
        with open(CHAT_HISTORY_FILE) as f:
            chats_db = {c["id"]: c for c in json.load(f).get("chats", [])}
    except Exception as e:
        logger.warning(f"Could not load chats: {e}")


# ─── FAISS indexing ───────────────────────────────────────────────────────────

def process_document_background(doc_id: str):
    """Extract → chunk → embed → update FAISS. Runs in thread pool."""
    global _vectorstore, _retriever

    with _processing_lock:
        doc_status[doc_id] = "indexing"
        try:
            data = documents_db.get(doc_id)
            if not data:
                doc_status[doc_id] = "error"
                return

            path = data["file_path"]
            ext  = data["file_type"]

            processors = {"pdf": process_pdf, "docx": process_docx, "txt": process_txt}
            text = processors[ext](path)
            documents_db[doc_id]["text_content"] = text

            if not text.strip():
                logger.warning(f"No text extracted from {doc_id}")
                doc_status[doc_id] = "error"
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". "],
            )
            chunks = splitter.split_text(text)[:MAX_CHUNKS]
            logger.info(f"{doc_id}: {len(chunks)} chunks")

            docs = []
            for i, chunk in enumerate(chunks):
                page_num = extract_page_number(chunk)
                docs.append(Document(
                    page_content=strip_page_markers(chunk),
                    metadata={
                        "doc_id":   doc_id,
                        "doc_name": data["filename"],
                        "chunk":    i,
                        "page":     page_num if page_num > 0 else (i // 5) + 1,
                    },
                ))

            emb   = get_embeddings()
            index = None
            for i in range(0, len(docs), EMBEDDING_BATCH_SIZE):
                batch = docs[i:i + EMBEDDING_BATCH_SIZE]
                new_idx = FAISS.from_documents(batch, emb)
                if index is None:
                    index = new_idx
                else:
                    index.merge_from(new_idx)
                time.sleep(0.3)

            if index is not None:
                if _vectorstore is None:
                    _vectorstore = index
                else:
                    _vectorstore.merge_from(index)
                _vectorstore.save_local(INDEX_DIR)
                _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})

            doc_status[doc_id] = "ready"
            save_documents_meta()
            logger.info(f"{doc_id} indexed successfully")

        except Exception as e:
            logger.error(f"Background processing failed for {doc_id}: {e}")
            doc_status[doc_id] = "error"


def rebuild_index():
    """Rebuild entire FAISS from all documents (used after delete)."""
    global _vectorstore, _retriever

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". "],
    )
    all_docs = []
    for doc_id, data in documents_db.items():
        text = data.get("text_content", "")
        if not text.strip():
            # Re-extract if text not in memory
            try:
                ext = data["file_type"]
                processors = {"pdf": process_pdf, "docx": process_docx, "txt": process_txt}
                text = processors[ext](data["file_path"])
                documents_db[doc_id]["text_content"] = text
            except Exception:
                continue
        for i, chunk in enumerate(splitter.split_text(text)[:MAX_CHUNKS]):
            all_docs.append(Document(
                page_content=strip_page_markers(chunk),
                metadata={"doc_id": doc_id, "doc_name": data["filename"], "chunk": i},
            ))

    if not all_docs:
        _vectorstore = None
        _retriever   = None
        return

    emb   = get_embeddings()
    index = None
    for i in range(0, len(all_docs), EMBEDDING_BATCH_SIZE):
        batch = all_docs[i:i + EMBEDDING_BATCH_SIZE]
        new_idx = FAISS.from_documents(batch, emb)
        index = new_idx if index is None else (index.merge_from(new_idx) or index)

    _vectorstore = index
    _vectorstore.save_local(INDEX_DIR)
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    logger.info(f"Index rebuilt with {len(all_docs)} chunks")


# ─── App lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_chats()
    load_documents_meta()
    logger.info("Loading embeddings model...")
    get_embeddings()

    global _vectorstore, _retriever
    idx_path = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(idx_path):
        try:
            _vectorstore = FAISS.load_local(
                INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True
            )
            _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            logger.info("FAISS index loaded from disk")
        except Exception as e:
            logger.warning(f"Could not load saved index: {e}")
    elif os.path.exists("faiss_index"):
        try:
            _vectorstore = FAISS.load_local(
                "faiss_index", get_embeddings(), allow_dangerous_deserialization=True
            )
            _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            logger.info("Legacy faiss_index loaded")
        except Exception as e:
            logger.warning(f"Could not load legacy index: {e}")

    yield
    save_chats()
    save_documents_meta()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Hybrid RAG System", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class SourceDoc(BaseModel):
    content:       str
    document_name: str
    chunk_id:      int
    page:          int = 0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    chat_id:  Optional[str] = None


class QueryResponse(BaseModel):
    answer:      str
    sources:     List[SourceDoc]
    question:    str
    chat_id:     Optional[str] = None
    source_type: str = "general"   # "documents" | "general" | "web"
    route:       str = "general"   # same values, used by frontend for badge


class DocResponse(BaseModel):
    id:          str
    filename:    str
    file_type:   str
    uploaded_at: str
    text_length: int
    status:      str = "ready"


class DocStatusResponse(BaseModel):
    id:     str
    status: str   # "pending" | "indexing" | "ready" | "error"


class ChatResp(BaseModel):
    id:         str
    title:      str
    created_at: str
    messages:   List[dict]


# ─── Route handlers ───────────────────────────────────────────────────────────

def _save_to_chat(chat_id: str, question: str, answer: str, sources: list, route: str = "general"):
    if chat_id and chat_id in chats_db:
        ts = datetime.now().isoformat()
        chats_db[chat_id]["messages"].extend([
            {"role": "user",      "content": question, "sources": None,    "timestamp": ts, "route": "user"},
            {"role": "assistant", "content": answer,   "sources": sources, "timestamp": ts, "route": route},
        ])
        if chats_db[chat_id].get("title", "New Chat") == "New Chat":
            chats_db[chat_id]["title"] = question[:50]
        save_chats()


def answer_from_documents(question: str, chat_id: str = None) -> tuple[str, List[SourceDoc]]:
    """Non-streaming RAG answer. Used by /query fallback."""
    docs, src_or_err = _retrieve_docs(question, chat_id=chat_id)
    if docs is None:
        return src_or_err, []

    q_lower     = question.lower()
    raw_context = format_docs(docs)
    context     = raw_context[:MAX_CONTEXT_CHARS]

    broad_triggers = [
        "about", "summarize", "summary", "overview", "describe",
        "what is", "what are", "explain", "tell me", "give me",
        "introduction", "intro", "topic", "contents", "main", "key points",
        "features", "analysis", "improvements", "detail", "in detail",
    ]
    is_broad = any(t in q_lower for t in broad_triggers)

    # Label for which document(s) we're answering from
    unique_docs = {d.metadata.get("doc_name", "") for d in docs}
    if len(unique_docs) == 1:
        doc_label = f'"{next(iter(unique_docs))}"'
    else:
        doc_label = "the uploaded documents"

    if is_broad:
        prompt = (
            f"IMPORTANT: Answer ONLY from the document excerpts below. "
            f"Do NOT use outside knowledge. Do NOT hallucinate. "
            f"The excerpts are from {doc_label}.\n\n"
            f"---BEGIN DOCUMENT EXCERPTS---\n{context}\n---END DOCUMENT EXCERPTS---\n\n"
            f"Question: {question}\n"
            f"Answer (based strictly on the excerpts above):"
        )
    else:
        prompt = (
            f"IMPORTANT: Answer ONLY from the document excerpts below. "
            f"Do NOT use outside knowledge. Do NOT hallucinate. "
            f"If the answer is not in the excerpts, say so clearly.\n\n"
            f"---BEGIN DOCUMENT EXCERPTS---\n{context}\n---END DOCUMENT EXCERPTS---\n\n"
            f"Question: {question}\n"
            f"Answer (based strictly on the excerpts above):"
        )

    resp   = get_llm().invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, src_or_err


def answer_general(question: str) -> str:
    """Direct LLM answer — no retrieval overhead."""
    prompt = (
        "You are Hybrid RAG, a knowledgeable AI assistant. "
        "Answer the question directly without any greeting or preamble. "
        "For technical or conceptual questions, explain clearly with key ideas and examples. "
        "For greetings, respond warmly in one sentence and ask how you can help. "
        "Never start your answer with 'Hello', 'Hi', or 'Hey'.\n\n"
        f"Question: {question}\nAnswer:"
    )
    resp = get_llm().invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def answer_web(question: str) -> str:
    """DuckDuckGo search → LLM summarises results."""
    try:
        results = get_web_search().run(question)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return answer_general(question)

    # Trim web results to avoid huge prompts
    trimmed = results[:2000]
    prompt  = (
        f"Answer the question using these web results. Be concise.\n\n"
        f"Results:\n{trimmed}\n\nQuestion: {question}\nAnswer:"
    )
    resp = get_llm().invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


from fastapi.responses import StreamingResponse
import json as _json


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming version of /query — sends tokens as they arrive via NDJSON."""
    loop = asyncio.get_event_loop()

    # Classify (mostly rule-based, fast)
    route = await loop.run_in_executor(None, classify_query, req.question)
    logger.info(f"Stream query: {route} | '{req.question[:60]}'")

    # Map classifier labels → frontend route keys (must match ROUTE_CONFIG in App.jsx)
    ROUTE_MAP = {"DOCUMENT": "documents", "WEB": "web", "GENERAL": "general"}
    frontend_route = ROUTE_MAP.get(route, "general")

    async def generate():
        # Send correct route label immediately so badge renders right away
        yield _json.dumps({"type": "route", "route": frontend_route}) + "\n"

        sources:     List[SourceDoc] = []
        source_type: str             = frontend_route
        full_answer: list            = []

        try:
            if route == "DOCUMENT":
                docs_result      = await loop.run_in_executor(
                    None, lambda: _retrieve_docs(req.question, chat_id=req.chat_id)
                )
                docs, src_or_err = docs_result

                if docs is None:
                    # src_or_err is an error string — emit as token
                    yield _json.dumps({"type": "token", "text": src_or_err}) + "\n"
                    full_answer.append(src_or_err)
                else:
                    sources = src_or_err   # List[SourceDoc]
                    async for token in _stream_doc_answer(req.question, docs):
                        yield _json.dumps({"type": "token", "text": token}) + "\n"
                        full_answer.append(token)

            elif route == "WEB":
                try:
                    web_results = await loop.run_in_executor(
                        None, lambda: get_web_search().run(req.question)
                    )
                    trimmed = web_results[:2000]
                except Exception as e:
                    logger.warning(f"Web search error: {e}")
                    trimmed = ""

                if trimmed:
                    prompt = (
                        "You are a helpful assistant. Answer the question accurately "
                        f"using these web search results.\n\nWeb Results:\n{trimmed}\n\n"
                        f"Question: {req.question}\nAnswer:"
                    )
                else:
                    prompt = (
                        "You are Hybrid RAG, a knowledgeable AI assistant. "
                        "Answer the question directly without greeting or preamble. "
                        "Never start with 'Hello', 'Hi', or 'Hey'.\n\n"
                        f"Question: {req.question}\nAnswer:"
                    )
                async for token in _stream_llm(prompt):
                    yield _json.dumps({"type": "token", "text": token}) + "\n"
                    full_answer.append(token)

            else:  # GENERAL
                prompt = (
                    "You are Hybrid RAG, a knowledgeable AI assistant. "
                    "Answer the question directly without any greeting or preamble. "
                    "For technical or conceptual questions, explain clearly with key ideas and examples. "
                    "For greetings, respond warmly in one sentence and ask how you can help. "
                    "Never start your answer with 'Hello', 'Hi', or 'Hey'.\n\n"
                    f"Question: {req.question}\nAnswer:"
                )
                async for token in _stream_llm(prompt):
                    yield _json.dumps({"type": "token", "text": token}) + "\n"
                    full_answer.append(token)

        except Exception as e:
            logger.error(f"Stream error: {e}")
            err_text = f"Sorry, an error occurred: {str(e)}"
            yield _json.dumps({"type": "token", "text": err_text}) + "\n"
            full_answer.append(err_text)

        # Save complete answer to chat history
        _save_to_chat(
            req.chat_id,
            req.question,
            "".join(full_answer),
            [s.model_dump() for s in sources],
            route=frontend_route,
        )

        # Signal stream complete
        yield _json.dumps({
            "type":        "done",
            "sources":     [s.model_dump() for s in sources],
            "source_type": source_type,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _retrieve_docs(question: str, chat_id: str = None):
    """
    Core retrieval. Uses active_doc session to remember which doc the user
    is currently working on — prevents cross-doc contamination on follow-ups.
    Returns (docs, List[SourceDoc]) on success, or (None, error_str) on failure.
    """
    global _active_doc

    if _vectorstore is None:
        pending = [d for d, s in doc_status.items() if s in ("pending", "indexing")]
        if pending:
            return None, "⏳ Your document is still being indexed. Please wait a moment and try again."
        return None, "No documents have been indexed yet. Please upload a document first."

    q_lower = question.lower()

    # ── Step 1: Identify target document ──────────────────────────────────────
    target_doc_id   = None
    target_doc_name = None

    # A) Explicit filename mentioned in question (highest priority)
    for doc_id, data in documents_db.items():
        fname      = data.get("filename", "")
        fname_stem = fname.rsplit(".", 1)[0].lower()
        if (fname_stem and len(fname_stem) > 3 and fname_stem in q_lower) \
           or fname.lower() in q_lower:
            target_doc_id   = doc_id
            target_doc_name = fname
            break

    # B) Vague doc reference → use most recently uploaded (and set as active)
    VAGUE_REFS = [
        "this doc", "my doc", "the doc", "this pdf", "my pdf", "the pdf",
        "this file", "my file", "this document", "the document",
        "this paper", "my paper", "the paper",
        "this report", "the report", "this assignment",
        "this cv", "my cv", "the cv", "this resume", "my resume",
    ]
    if target_doc_id is None and any(t in q_lower for t in VAGUE_REFS):
        if len(documents_db) == 1:
            target_doc_id   = list(documents_db.keys())[0]
            target_doc_name = list(documents_db.values())[0].get("filename", "")
        elif len(documents_db) > 1:
            sorted_docs   = sorted(documents_db.items(),
                                   key=lambda x: x[1].get("uploaded_at", ""), reverse=True)
            target_doc_id   = sorted_docs[0][0]
            target_doc_name = sorted_docs[0][1].get("filename", "")

    # C) No explicit target — check active_doc session for follow-up questions
    if target_doc_id is None and chat_id and chat_id in _active_doc:
        saved_id = _active_doc[chat_id]
        if saved_id in documents_db:
            target_doc_id   = saved_id
            target_doc_name = documents_db[saved_id].get("filename", "")

    # D) Only one doc uploaded → always use it
    if target_doc_id is None and len(documents_db) == 1:
        target_doc_id   = list(documents_db.keys())[0]
        target_doc_name = list(documents_db.values())[0].get("filename", "")

    # ── Step 2: Remember active doc for this chat session ─────────────────────
    if target_doc_id and chat_id:
        _active_doc[chat_id] = target_doc_id

    # ── Step 3: Broad vs specific — determines how many chunks to fetch ────────
    broad_triggers = [
        "about", "summarize", "summary", "overview", "describe",
        "explain", "tell me", "give me", "introduction", "intro",
        "topic", "contents", "main", "key points", "features",
        "analysis", "detail", "in detail", "in short", "briefly",
        "skills", "experience", "certifications", "education",
        "findings", "results", "methodology", "conclusion",
    ]
    is_broad = any(t in q_lower for t in broad_triggers)
    want_k   = TOP_K_BROAD if is_broad else TOP_K

    # ── Step 4: FAISS retrieval ────────────────────────────────────────────────
    # When filtering by a specific doc, fetch many more candidates — the target
    # doc's chunks may not rank in the global top-K for generic queries
    fetch_k = 50 if target_doc_id else want_k * 2
    results = _vectorstore.similarity_search_with_relevance_scores(question, k=fetch_k)

    # Filter to target doc if we have one
    if target_doc_id:
        filtered = [(d, s) for d, s in results
                    if d.metadata.get("doc_id") == target_doc_id]
        if filtered:
            results = filtered
        else:
            # Doc IS indexed (status=ready) but didn't appear in top-50 similarity results.
            # Fall back to a direct doc-filtered search with lower k constraint.
            try:
                all_results = _vectorstore.similarity_search_with_relevance_scores(
                    question, k=_vectorstore.index.ntotal
                )
                filtered = [(d, s) for d, s in all_results
                            if d.metadata.get("doc_id") == target_doc_id]
                if filtered:
                    results = filtered
                else:
                    status = doc_status.get(target_doc_id, "unknown")
                    if status in ("pending", "indexing"):
                        return None, f"⏳ '{target_doc_name}' is still being indexed. Please wait and try again."
                    return None, f"No content found for '{target_doc_name}'. Try re-uploading the file."
            except Exception:
                status = doc_status.get(target_doc_id, "unknown")
                if status in ("pending", "indexing"):
                    return None, f"⏳ '{target_doc_name}' is still being indexed. Please wait and try again."
                return None, f"No content found for '{target_doc_name}'. Try re-uploading the file."

    # ── Step 5: Relevance filtering ───────────────────────────────────────────
    good = [(d, s) for d, s in results if s >= RELEVANCE_THRESHOLD]

    if not good:
        if is_broad and results:
            # For broad questions, take best available chunks even if below threshold
            good = results[:want_k]
        elif results and results[0][1] >= 0.20:
            # Specific question: only use if somewhat relevant
            good = results[:2]
        else:
            return None, "I could not find relevant information in your uploaded documents for this question."

    good = good[:want_k]
    docs = [d for d, _ in good]
    sources = [
        SourceDoc(
            content=d.page_content,
            document_name=d.metadata.get("doc_name", "Unknown"),
            chunk_id=d.metadata.get("chunk", 0),
            page=d.metadata.get("page", 0),
        )
        for d in docs
    ]
    return docs, sources


async def _stream_llm(prompt: str):
    """Async generator that yields tokens from Ollama astream."""
    llm = get_llm()
    async for chunk in llm.astream(prompt):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if text:
            yield text


async def _stream_doc_answer(question: str, docs):
    """Stream the LLM answer for document queries."""
    q_lower     = question.lower()
    context     = format_docs(docs)[:MAX_CONTEXT_CHARS]

    broad_triggers = [
        "about", "summarize", "summary", "overview", "describe",
        "what is", "what are", "explain", "tell me", "give me",
        "introduction", "intro", "topic", "contents", "main", "key points",
        "features", "analysis", "improvements", "detail", "in detail",
    ]
    is_broad = any(t in q_lower for t in broad_triggers)

    unique_docs = {d.metadata.get("doc_name", "") for d in docs}
    doc_label   = f'"{next(iter(unique_docs))}"' if len(unique_docs) == 1 else "the uploaded documents"

    if is_broad:
        prompt = (
            f"IMPORTANT: Answer ONLY from the document excerpts below. "
            f"Do NOT use outside knowledge. Do NOT hallucinate. "
            f"The excerpts are from {doc_label}.\n\n"
            f"---BEGIN DOCUMENT EXCERPTS---\n{context}\n---END DOCUMENT EXCERPTS---\n\n"
            f"Question: {question}\n"
            f"Answer (based strictly on the excerpts above):"
        )
    else:
        prompt = (
            f"IMPORTANT: Answer ONLY from the document excerpts below. "
            f"Do NOT use outside knowledge. Do NOT hallucinate. "
            f"If the answer is not in the excerpts, say so clearly.\n\n"
            f"---BEGIN DOCUMENT EXCERPTS---\n{context}\n---END DOCUMENT EXCERPTS---\n\n"
            f"Question: {question}\n"
            f"Answer (based strictly on the excerpts above):"
        )

    async for token in _stream_llm(prompt):
        yield token




@app.get("/health")
async def health():
    return {
        "status": "ok",
        "docs":   len(documents_db),
        "chats":  len(chats_db),
        "indexed": _vectorstore is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Main query endpoint — classifies then routes to correct handler."""
    loop  = asyncio.get_event_loop()

    # Step 1: classify
    route = await loop.run_in_executor(None, classify_query, req.question)
    logger.info(f"Query classified as: {route} | '{req.question[:60]}'")

    ROUTE_MAP   = {"DOCUMENT": "documents", "WEB": "web", "GENERAL": "general"}
    source_type = ROUTE_MAP.get(route, "general")
    sources: List[SourceDoc] = []

    # Step 2: route
    if route == "DOCUMENT":
        answer, sources = await loop.run_in_executor(
            None, answer_from_documents, req.question, req.chat_id
        )
    elif route == "WEB":
        answer = await loop.run_in_executor(None, answer_web, req.question)
    else:
        answer = await loop.run_in_executor(None, answer_general, req.question)

    # Step 3: save to chat history
    _save_to_chat(
        req.chat_id,
        req.question,
        answer,
        [s.model_dump() for s in sources],
        route=source_type,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=req.question,
        chat_id=req.chat_id,
        source_type=source_type,
        route=source_type,
    )


@app.post("/documents/upload", response_model=DocResponse)
async def upload(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ["pdf", "docx", "txt"]:
        raise HTTPException(400, f"Unsupported file type: .{ext}. Allowed: pdf, docx, txt")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large. Maximum is 10 MB.")

    # Quick PDF page count guard
    if ext == "pdf":
        try:
            import pymupdf
            doc = pymupdf.open(stream=content, filetype="pdf")
            pages = len(doc)
            doc.close()
            if pages > MAX_PAGES:
                raise HTTPException(413, f"PDF has {pages} pages. Maximum is {MAX_PAGES}.")
        except HTTPException:
            raise
        except Exception:
            pass

    doc_id = str(uuid.uuid4())
    path   = os.path.join(STORAGE_DIR, f"{doc_id}.{ext}")

    try:
        with open(path, "wb") as f:
            f.write(content)

        documents_db[doc_id] = {
            "id":           doc_id,
            "filename":     file.filename,
            "file_type":    ext,
            "uploaded_at":  datetime.now().isoformat(),
            "text_content": "",
            "file_path":    path,
        }
        doc_status[doc_id] = "pending"

        # Non-blocking background indexing
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, process_document_background, doc_id)

        return DocResponse(
            id=doc_id,
            filename=file.filename,
            file_type=ext,
            uploaded_at=documents_db[doc_id]["uploaded_at"],
            text_length=0,
            status="pending",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(500, str(e))


@app.get("/documents", response_model=List[DocResponse])
async def list_docs():
    return [
        DocResponse(
            id=d["id"],
            filename=d["filename"],
            file_type=d["file_type"],
            uploaded_at=d["uploaded_at"],
            text_length=len(d.get("text_content", "")),
            status=doc_status.get(d["id"], "ready"),
        )
        for d in documents_db.values()
    ]


@app.get("/documents/{doc_id}/status", response_model=DocStatusResponse)
async def get_doc_status(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(404, "Document not found")
    return DocStatusResponse(id=doc_id, status=doc_status.get(doc_id, "ready"))


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(404, "Document not found")

    path = documents_db[doc_id].get("file_path")
    if path and os.path.exists(path):
        os.remove(path)

    del documents_db[doc_id]
    if doc_id in doc_status:
        del doc_status[doc_id]

    # Rebuild index in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, rebuild_index)
    save_documents_meta()
    return {"message": "Deleted"}


# ─── Chat endpoints ───────────────────────────────────────────────────────────

@app.get("/chats", response_model=List[ChatResp])
async def list_chats():
    return [
        ChatResp(
            id=c["id"],
            title=c.get("title", "New Chat"),
            created_at=c.get("created_at", ""),
            messages=c.get("messages", []),
        )
        for c in sorted(chats_db.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    ]


@app.post("/chats", response_model=ChatResp)
async def create_chat():
    chat_id = str(uuid.uuid4())
    chats_db[chat_id] = {
        "id":         chat_id,
        "title":      "New Chat",
        "created_at": datetime.now().isoformat(),
        "messages":   [],
    }
    save_chats()
    return ChatResp(**chats_db[chat_id])


@app.get("/chats/{chat_id}", response_model=ChatResp)
async def get_chat(chat_id: str):
    if chat_id not in chats_db:
        raise HTTPException(404, "Chat not found")
    return ChatResp(**chats_db[chat_id])


@app.patch("/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, req: dict):
    if chat_id not in chats_db:
        raise HTTPException(404, "Chat not found")
    if "title" not in req:
        raise HTTPException(400, "Title required")
    chats_db[chat_id]["title"] = req["title"]
    save_chats()
    return {"id": chat_id, "title": req["title"]}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id not in chats_db:
        raise HTTPException(404, "Chat not found")
    del chats_db[chat_id]
    save_chats()
    return {"message": "Deleted"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)