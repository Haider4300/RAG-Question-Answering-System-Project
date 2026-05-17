# 🧠 Hybrid RAG AI System

An intelligent **Hybrid Retrieval-Augmented Generation (RAG) AI System** that dynamically routes user queries between:

- 📄 Document-based Retrieval (RAG)
- 💡 General AI Responses
- 🌐 Real-Time Web Search

This project combines **semantic retrieval**, **LLMs**, **vector search**, and **live web search** into a single AI assistant capable of handling both static and dynamic knowledge sources.


---

# 🚀 Project Highlights

✅ Hybrid AI Routing Architecture  
✅ Semantic Document Search using FAISS  
✅ Multi-Document Question Answering  
✅ Real-Time Web Search Integration  
✅ Streaming LLM Responses  
✅ Context-Aware Conversations  
✅ FastAPI Backend + React Frontend  
✅ Ollama Local LLM Integration  
✅ Persistent Multi-Chat Sessions  
✅ Source Citations with Page Numbers  

---

# 📌 Why This Project?

Traditional chatbots struggle to determine:

- When to retrieve information from documents
- When to answer directly using LLM knowledge
- When live/current web information is required

This project solves that problem using an intelligent routing system that dynamically selects the best response pipeline.

---

# 🏗️ System Architecture

```text
                          ┌────────────────────┐
                          │    User Query      │
                          └─────────┬──────────┘
                                    │
                                    ▼
                     ┌─────────────────────────┐
                     │ Intelligent Query Router│
                     │  (Rule-Based Routing)  │
                     └─────────┬──────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼

 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │  Document RAG  │   │  General LLM   │   │   Web Search   │
 └───────┬────────┘   └────────┬───────┘   └────────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼

 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │ FAISS VectorDB │   │ Gemma3 via     │   │ DuckDuckGo     │
 │ Semantic Search│   │ Ollama         │   │ Live Search    │
 └───────┬────────┘   └────────┬───────┘   └────────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼

 ┌──────────────────────────────────────────────────────────┐
 │                 Final AI Response                       │
 └──────────────────────────────────────────────────────────┘
```

---

# ⚡ How Query Routing Works

The system intelligently determines which pipeline should handle the query.

---

## 📄 Route 1 — Document RAG

Triggered when:
- User asks about uploaded documents
- User references PDFs/files
- Follow-up questions require document context

### Features
- Semantic vector search
- Multi-document retrieval
- Source citations
- Context-aware follow-ups
- Page-level referencing

### Example Queries

```text
"Summarize this PDF"
"What are the financial results?"
"Explain chapter 3 from the uploaded document"
```

---

## 💡 Route 2 — General LLM

Triggered when:
- User asks conceptual/general knowledge questions
- No uploaded document context exists

### Features
- Direct LLM response
- Fast inference
- Coding assistance
- Explanations
- AI/ML help

### Example Queries

```text
"What is deep learning?"
"Explain transformers"
"Write a Python sorting function"
```

---

## 🌐 Route 3 — Web Search

Triggered when:
- Query requires current/live information

### Features
- Real-time search
- News retrieval
- Weather updates
- Recent events
- Current AI trends

### Example Queries

```text
"Latest GPT-5 news"
"Weather in Lahore"
"Who won yesterday's match?"
```

---

# 🧩 Core AI Engineering Concepts

This project demonstrates multiple modern AI engineering patterns used in real-world AI systems:

| Concept | Description |
|---|---|
| Retrieval-Augmented Generation (RAG) | Combines retrieval with generation |
| Semantic Search | Meaning-based retrieval |
| Vector Similarity Search | FAISS embedding search |
| Query Classification | Intelligent route selection |
| Context Injection | Injecting retrieved chunks into prompts |
| Streaming Responses | Token-by-token generation |
| Multi-Pipeline AI Systems | Combining multiple AI routes |
| Conversational Context Tracking | Persistent chat context |
| Hybrid Knowledge Systems | Static + dynamic knowledge |

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM Runtime | Ollama |
| LLM Model | gemma3:4b |
| Embeddings | nomic-embed-text |
| Vector Database | FAISS |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | React + Vite |
| Styling | Tailwind CSS |
| Web Search | DuckDuckGo |
| Streaming | NDJSON |
| Language | Python |

---

# 📂 Project Structure

```text
Hybrid-RAG-System/
│
├── api.py
├── main.py
├── pyproject.toml
├── README.md
│
├── uploads/
├── vector_store/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   ├── lib/
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
└── screenshots/
```

---

# 📸 Screenshots

## 💬 Chat Interface

```text
Add screenshot:
screenshots/chat-ui.png
```

---

## 📄 Document Upload

```text
Add screenshot:
screenshots/upload.png
```

---

## ⚡ Streaming Responses

```text
Add screenshot:
screenshots/streaming.png
```

---

## 📚 Source Citations

```text
Add screenshot:
screenshots/citations.png
```

---

# ⚙️ Installation & Setup

---

## 1️⃣ Prerequisites

Install the following:

- Python 3.11+
- Node.js 18+
- Ollama

Download Ollama:

```text
https://ollama.ai/
```

---

# 2️⃣ Pull Required Models

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

---

# 3️⃣ Install Backend Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install fastapi uvicorn langchain langchain-community \
langchain-ollama faiss-cpu pymupdf python-docx \
python-multipart duckduckgo-search
```

---

# 4️⃣ Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

# ▶️ Run The Application

---

## Backend

```bash
python api.py
```

Runs on:

```text
http://localhost:8000
```

---

## Frontend

```bash
cd frontend
npm run dev
```

Runs on:

```text
http://localhost:5173
```

---

# 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | Standard query |
| POST | `/query/stream` | Streaming query |
| POST | `/documents/upload` | Upload document |
| GET | `/documents` | List documents |
| DELETE | `/documents/{id}` | Delete document |
| GET | `/chats` | List chats |
| POST | `/chats` | Create chat |
| GET | `/chats/{id}` | Get chat history |

---

# 🌐 Deployment Options

| Platform | Usage |
|---|---|
| Hugging Face Spaces | Full AI app deployment |
| Vercel | Frontend hosting |
| Render | Backend hosting |

---


# 📈 Current Capabilities

✅ Hybrid AI Routing  
✅ Semantic Document Retrieval  
✅ Streaming LLM Responses  
✅ Persistent Conversations  
✅ Context-Aware Follow-Ups  
✅ Real-Time Web Search  
✅ Multi-Document Support  
✅ Full-Stack AI Architecture  

---

# 🚀 Future Improvements

Planned future enhancements:

- Docker deployment
- Redis caching
- PostgreSQL integration
- Authentication system
- Hybrid dense+sparse retrieval
- LangGraph workflows
- Multi-modal document support
- Voice assistant integration
- GPU inference optimization
- Agentic tool calling
- Memory-enhanced conversations

---

# 🧪 Example Workflow

```text
User uploads PDF
        │
        ▼
Document indexed into FAISS
        │
        ▼
User asks question
        │
        ▼
Query Router detects DOCUMENT route
        │
        ▼
Relevant chunks retrieved
        │
        ▼
Chunks injected into LLM prompt
        │
        ▼
LLM generates contextual answer
        │
        ▼
Answer returned with citations
```

---

# 👨‍💻 Author

## Ali Haider 

AI Engineer 

Focused on:
- Retrieval-Augmented Generation (RAG)
- AI Engineering
- Machine Learning
- NLP Systems
- Full-Stack AI Applications
- Computer Vision

---

# 📜 License

Internal project. All rights reserved by the author.

---
