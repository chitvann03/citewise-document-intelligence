from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="CiteWise", version="0.1.0")


@dataclass
class Chunk:
    text: str
    document_name: str
    page: int


class DocumentStore:
    """A deliberately small in-memory retrieval index for the portfolio MVP."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def add_pdf(self, filename: str, content: bytes) -> int:
        reader = PdfReader(BytesIO(content))
        new_chunks: list[Chunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk in split_into_chunks(text):
                new_chunks.append(Chunk(chunk, filename, page_number))

        if not new_chunks:
            raise ValueError("No readable text was found. The PDF may be a scanned image.")

        # Re-uploading a document should refresh its index, not duplicate it.
        self.chunks = [chunk for chunk in self.chunks if chunk.document_name != filename]
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def search(self, question: str, limit: int = 3) -> list[tuple[Chunk, float]]:
        if self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).flatten()
        ranked_indexes = scores.argsort()[::-1]
        selected: list[tuple[Chunk, float]] = []
        for index in ranked_indexes:
            candidate = self.chunks[index]
            # Overlapping chunks often contain almost the same text. Returning
            # all of them looks like duplicate evidence, so keep diverse passages.
            if any(not is_distinct(candidate.text, chosen.text) for chosen, _ in selected):
                continue
            selected.append((candidate, float(scores[index])))
            if len(selected) == limit:
                break
        return selected

    def _rebuild_index(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])


store = DocumentStore()


def split_into_chunks(text: str, words_per_chunk: int = 110, overlap: int = 25) -> list[str]:
    """Split text into overlapping word windows to preserve nearby context."""
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
    """Reject near-duplicate chunks produced by overlapping chunk windows."""
    first_words = set(re.findall(r"\w+", first.lower()))
    second_words = set(re.findall(r"\w+", second.lower()))
    if not first_words or not second_words:
        return True
    overlap = len(first_words & second_words) / len(first_words | second_words)
    return overlap < maximum_overlap


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


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
    if not results or results[0][1] < 0.15:
        return {
            "answer": "I don't know based on the uploaded documents.",
            "hint": "Try a more specific question using terms from the document.",
            "sources": [],
            "grounded": False,
        }

    sources = [
        {
            "document": chunk.document_name,
            "page": chunk.page,
            "excerpt": chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""),
            "score": round(score, 3),
        }
        for chunk, score in results
    ]
    return {
        "answer": "Here is the strongest evidence I found. Verify the cited page before relying on it.",
        "sources": sources,
        "grounded": True,
    }


HTML = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CiteWise</title><style>
:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#0c1020; color:#edf0ff; }
body { max-width:850px; margin:0 auto; padding:48px 22px; } h1 { font-size:clamp(2rem,6vw,3.5rem); margin:0; }
.sub { color:#aeb8d9; line-height:1.6; } .card { background:#151b33; border:1px solid #2b365e; border-radius:16px; padding:22px; margin:22px 0; }
input, textarea, button { font:inherit; border-radius:9px; padding:12px; } input, textarea { width:100%; box-sizing:border-box; background:#0c1020; border:1px solid #3b4877; color:white; }
textarea { min-height:105px; resize:vertical; } button { border:0; cursor:pointer; background:#8ba4ff; color:#0a1025; font-weight:700; margin-top:10px; } button:hover { background:#b4c3ff; }
#status { color:#b9c6ed; min-height:1.5em; } .source { border-left:3px solid #8ba4ff; padding:10px 14px; margin:12px 0; background:#0c1020; border-radius:0 8px 8px 0; }
.meta { color:#aeb8d9; font-size:.86rem; } .answer { line-height:1.6; font-weight:600; }
</style></head><body><h1>CiteWise</h1><p class="sub">Ask questions about your documents. Answers include page-level evidence—or a transparent “I don’t know.”</p>
<section class="card"><h2>1. Upload a PDF</h2><input id="file" type="file" accept="application/pdf"><button onclick="upload()">Index document</button><p id="status"></p></section>
<section class="card"><h2>2. Ask a question</h2><textarea id="question" placeholder="What is the leave policy?"></textarea><button onclick="ask()">Find cited answer</button><div id="result"></div></section>
<script>
const status = document.querySelector('#status'), result = document.querySelector('#result');
async function upload() { const file = document.querySelector('#file').files[0]; if (!file) return status.textContent='Choose a PDF first.'; status.textContent='Extracting and indexing…'; const data=new FormData(); data.append('file',file); const r=await fetch('/api/documents',{method:'POST',body:data}); const j=await r.json(); status.textContent=r.ok ? j.message : j.detail; }
async function ask() { const question=document.querySelector('#question').value.trim(); if (!question) return; result.innerHTML='<p class="meta">Searching documents…</p>'; const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})}); const j=await r.json(); if(!r.ok) return result.textContent=j.detail; result.innerHTML=`<p class="answer">${j.answer}</p>${j.hint ? `<p class="meta">${j.hint}</p>` : ''}` + (j.sources||[]).map(s=>`<article class="source"><strong>${s.document} — page ${s.page}</strong><p>${s.excerpt}</p><span class="meta">Retrieval score: ${s.score}</span></article>`).join(''); }
</script></body></html>
"""
