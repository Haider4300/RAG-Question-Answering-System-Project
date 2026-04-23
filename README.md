# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions about Sarah Chen's CV using vector search (FAISS) and Ollama LLMs.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   doc.txt   │────▶│   Ollama    │────▶│    FAISS    │
│  (source)   │     │ embeddings  │     │   index     │
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

## Project Structure

```
rag/
├── main.py           # CLI version (one-time index + query)
├── api.py            # FastAPI server for inference
├── doc.txt           # Source document (Sarah Chen's CV)
├── faiss_index/      # Vector store (created at runtime)
│   ├── index.faiss
│   └── index.pkl
├── frontend/         # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── lib/api.js
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md
```

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

### 3. Install Python Dependencies

```bash
cd rag
pip install -e .
# or
uv sync
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Usage

### Option 1: API + Frontend (Recommended)

**Terminal 1 - Start FastAPI:**
```bash
cd rag
python api.py
```

**Terminal 2 - Start Frontend:**
```bash
cd rag/frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Option 2: CLI (Direct)

```bash
cd rag
python main.py
Enter your question: What is Sarah Chen's work experience?
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Ask a question |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Sarah Chen'\''s work experience?"}'
```

### Example Response

```json
{
  "answer": "Sarah Chen is a Senior Data Scientist at TechInnovate Solutions...",
  "sources": [
    {
      "content": "Senior Data Scientist\nTechInnovate Solutions, Toronto, ON\nJanuary 2021 – Present\n...",
      "metadata": {"source": "doc.txt"}
    }
  ],
  "question": "What is Sarah Chen's work experience?"
}
```

## Rebuilding the Index

If `doc.txt` changes, delete `faiss_index/` and either:

- Run `python main.py` once to rebuild, then use `api.py`
- Or modify `api.py` to rebuild on startup (add index creation before loading)

## Environment

- **Python**: 3.11+
- **Ollama**: Must be running (`ollama serve`)
- **Ports**: Frontend: 5173, API: 8000