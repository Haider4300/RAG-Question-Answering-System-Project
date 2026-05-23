# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend + serve frontend
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY api.py .
COPY doc.txt .

# Copy built frontend into static folder
COPY --from=frontend-builder /frontend/dist/ ./static/

# Create runtime directories
RUN mkdir -p uploaded_files faiss_index_uploaded

# HF Spaces runs on port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]