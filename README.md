# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system with document upload, chat history, and web search fallback. Upload PDF, DOCX, or TXT documents and ask questions about them using vector search (FAISS) and Ollama LLMs.

## Features

- **Document Upload**: Upload PDF, DOCX, TXT files (up to 10MB and 250 pages)
- **Chat History**: Multiple conversations with persistent storage
- **RAG-powered Answers**: Get answers based on your uploaded documents
- **Web Search Fallback**: Falls back to web search when no documents are uploaded
- **Responsive Design**: Works on desktop and mobile

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Uploaded  │────▶│   Ollama    │────▶│    FAISS    │
│  Documents  │     │ embeddings │     │   index     │
└─────────────┘     └─────────────┘     └─────────────┘
                                             │
                                             ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │◀────│   Ollama    │◀────│  Retriever  │
│    API      │     │   LLM       │     │   (top-k)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│   React     │
│   Frontend  │
└─────────────┘
```

## Tech Stack

- **Embeddings**: `nomic-embed-text` via Ollama
- **LLM**: `minimax-m2.7:cloud` via Ollama ChatOllama
- **Vector Store**: FAISS
- **RAG Framework**: LangChain
- **Backend**: FastAPI
- **Frontend**: React + Tailwind CSS + Vite

## Setup

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai/) installed and running

### 2. Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull minimax-m2.7:cloud
```

### 3. Install Dependencies

```bash
# Backend
pip install fastapi uvicorn langchain langchain-ollama langchain-community faiss-cpu pymupdf python-docx

# Frontend
cd frontend
npm install
```

## Usage

### Start FastAPI Backend

```bash
python api.py
```

### Start Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Document Upload Limits

- **Max file size**: 10 MB
- **Max pages**: 250 pages
- **Supported formats**: PDF, DOCX, TXT

For best results, upload smaller documents or split larger books into chapters.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Upload a document |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `POST` | `/query` | Ask a question |
| `GET` | `/chats` | List chat conversations |
| `POST` | `/chats` | Create new chat |
| `GET` | `/chats/{chat_id}` | Get chat messages |
| `DELETE` | `/chats/{chat_id}` | Delete chat |

## Project Structure

```
├── api.py                 # FastAPI backend server
├── main.py               # CLI version (one-time indexing)
├── chat_history.json     # Chat storage (created at runtime)
├── faiss_index_uploaded/ # Vector index (created at runtime)
├── uploaded_files/       # Uploaded documents (created at runtime)
├── frontend/             # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── lib/api.js
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Environment

- **Python**: 3.11+
- **Ollama**: Must be running (`ollama serve`)
- **Ports**: Frontend: 5173, API: 8000
