import logging
import re
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from app.answering import create_answer

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CiteWise", version="0.1.0")

# Setup template rendering
templates = Jinja2Templates(directory="app/templates")

class DocumentStore:
    def __init__(self) -> None:
        # Use a persistent client so the data survives restarts
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # This will download the sentence-transformer model on first run
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="citewise_docs",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"DocumentStore initialized. Collection count: {self.collection.count()}")

    def add_pdf(self, filename: str, content: bytes) -> int:
        logger.info(f"Processing PDF: {filename}")
        reader = PdfReader(BytesIO(content))
        chunks = []
        metadatas = []
        ids = []
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(split_into_chunks(text)):
                chunks.append(chunk)
                metadatas.append({"document_name": filename, "page": page_number})
                ids.append(f"{filename}_p{page_number}_{i}")

        if not chunks:
            raise ValueError("No readable text was found. The PDF may be a scanned image.")

        # Delete any existing chunks for this document so we don't duplicate
        try:
            self.collection.delete(where={"document_name": filename})
        except Exception as e:
            logger.warning(f"Could not delete old chunks for {filename}: {e}")

        # Add chunks to chromadb in batches to be safe
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                documents=chunks[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        
        logger.info(f"Successfully added {len(chunks)} chunks for {filename}")
        return len(chunks)

    def search(self, question: str, limit: int = 3) -> list[dict]:
        logger.info(f"Searching for: {question}")
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[question],
            n_results=limit * 2 # get extra for diversity filtering
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        selected = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: lower is better (0 is exact). 
            # We convert to similarity for the frontend.
            score = 1.0 - dist
            
            # Simple diversity check
            if any(not is_distinct(doc, chosen['text']) for chosen in selected):
                continue
                
            selected.append({
                "text": doc,
                "document_name": meta["document_name"],
                "page": meta["page"],
                "score": score
            })
            
            if len(selected) == limit:
                break
                
        return selected

store = DocumentStore()

def split_into_chunks(text: str, words_per_chunk: int = 110, overlap: int = 25) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    step = words_per_chunk - overlap
    for start in range(0, len(words), step):
        window = " ".join(words[start : start + words_per_chunk])
        if len(window) >= 40:
            chunks.append(window)
    return chunks

def is_distinct(first: str, second: str, maximum_overlap: float = 0.65) -> bool:
    first_words = set(re.findall(r"\w+", first.lower()))
    second_words = set(re.findall(r"\w+", second.lower()))
    if not first_words or not second_words:
        return True
    overlap = len(first_words & second_words) / len(first_words | second_words)
    return overlap < maximum_overlap

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    content = await file.read()
    try:
        count = store.add_pdf(file.filename, content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"message": f"Indexed {count} chunks from {file.filename}.", "chunks": count}

@app.post("/api/ask")
async def ask_question(payload: dict) -> dict:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")
    results = store.search(question)
    
    # Cosine distance similarity threshold
    if not results or results[0]["score"] < 0.2:
        logger.info(f"No confident match found. Best score: {results[0]['score'] if results else 'N/A'}")
        return {
            "answer": "I don't know based on the uploaded documents.",
            "hint": "Try a more specific question using terms from the document.",
            "sources": [],
            "grounded": False,
        }

    sources = [
        {
            "document": chunk["document_name"],
            "page": chunk["page"],
            "excerpt": chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else ""),
            "score": round(chunk["score"], 3),
        }
        for chunk in results
    ]
    answer, mode, notice = create_answer(question, sources)
    return {
        "answer": answer,
        "sources": sources,
        "grounded": True,
        "mode": mode,
        "notice": notice,
    }
