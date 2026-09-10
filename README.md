# CiteWise — Document Intelligence Assistant

CiteWise lets a user upload PDF documents and ask questions about them. Each answer is grounded in retrieved passages and shows the page-level sources used, so the app can say **"I don't know"** instead of inventing an answer.

## What this project demonstrates

- PDF text extraction and page-aware document processing
- Retrieval-augmented generation (RAG) architecture
- TF-IDF vector retrieval (runs locally; no paid API needed)
- FastAPI backend and browser-based UI
- Transparent citations, confidence thresholding, and simple evaluation hooks

## Architecture

`PDF upload → page text extraction → chunking → vector index → retrieve relevant chunks → answer with citations`

The starter uses extractive answers: it returns the most relevant source passages, rather than pretending an LLM generated a reliable answer. This makes it inexpensive and easy to explain. An LLM can later be added after retrieval, but it must be instructed to use only those passages.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Interview notes

- **Why RAG, not fine-tuning?** Documents change frequently; RAG updates the knowledge base without retraining a model.
- **Why citations?** They make answers auditable and help users verify claims.
- **What prevents hallucination?** Retrieval confidence thresholding and a refusal response when no relevant chunk is found.
- **Current limitation:** TF-IDF is lexical and can miss semantic matches. Replace it with embeddings and a vector database as the next version.

## Next upgrades

1. Swap TF-IDF for sentence-transformer embeddings and Chroma/FAISS.
2. Add an LLM synthesis step constrained to retrieved text.
3. Save documents and indexes in PostgreSQL/object storage.
4. Add authentication, evaluation dataset, and Docker deployment.
