# main.py
"""
Intellextion - FastAPI backend (Hugging Face local Q&A + Sentence-Transformers + FAISS)
Requirements (install in your venv):
    pip install fastapi uvicorn pydantic "passlib[bcrypt]" python-multipart
    pip install transformers sentence-transformers faiss-cpu PyPDF2 python-magic-binary
Notes:
 - Model downloads may be large and take time on first run (transformers & sentence-transformers).
 - This implementation builds embeddings at upload-time and stores them in the DB (pickled).
 - Q&A uses a Hugging Face `question-answering` pipeline (distilbert by default).
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple
import sqlite3
import hashlib
import uuid
import os
import logging
import pickle
import math

# ML
import numpy as np
import faiss
import PyPDF2

# Optional libmagic fallback
try:
    import magic as _magic  # type: ignore
    _HAS_MAGIC = True
except Exception:
    _magic = None  # type: ignore
    _HAS_MAGIC = False

# Transformers lazy imports — we'll import inside functions to avoid long startup where possible
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# -------------------------
# App & config
# -------------------------
app = FastAPI(title="Intellextion (HF Q&A)", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DATABASE_PATH = BASE_DIR / "intellextion.db"

SECRET_KEY = os.getenv("INTELLEXTION_SECRET", "change-this-secret-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24  # 30 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# -------------------------
# Helpers: DB
# -------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_database() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size INTEGER NOT NULL,
            file_hash TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'uploaded',
            summary TEXT,
            is_archived INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            chunk_index INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)
    # seed a default root user if not present
    cur.execute("SELECT id FROM users WHERE username = ?", ("root",))
    if not cur.fetchone():
        root_hash = pwd_context.hash("root")
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    ("root", "root@example.com", root_hash))
    conn.commit()
    conn.close()

# -------------------------
# Pydantic models
# -------------------------
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ForgotPassword(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class DocumentResponse(BaseModel):
    id: str
    filename: str
    size: int
    uploaded_at: datetime
    status: str
    summary: Optional[str] = None

class Question(BaseModel):
    question: str

# -------------------------
# Auth utilities
# -------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # type: ignore
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, email FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise credentials_exception
    return {"id": row[0], "username": row[1], "email": row[2]}

# -------------------------
# Utilities: files, PDF, hashing, chunking
# -------------------------
def calculate_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

def validate_pdf_file(file_content: bytes) -> bool:
    if _HAS_MAGIC and _magic is not None:
        try:
            mime = _magic.from_buffer(file_content, mime=True)
            if mime == "application/pdf":
                return True
        except Exception:
            pass
    return file_content.startswith(b"%PDF-")

def extract_pdf_text(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts: List[str] = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as e:
        logging.error(f"extract_pdf_text error: {e}")
        return ""

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Simple word-based chunker. max_tokens is approx words per chunk.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

# -------------------------
# ML: lazy model loaders & helpers
# -------------------------
# We instantiate the embedder and QA pipeline lazily on first use (to avoid long cold startup)
_EMBEDDER = None  # SentenceTransformer
_QA_PIPE = None   # transformers pipeline

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        logging.info("Loading sentence-transformers model (this may take a while)...")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("Embedder loaded.")
    return _EMBEDDER

def get_qa_pipeline():
    global _QA_PIPE
    if _QA_PIPE is None:
        logging.info("Loading QA pipeline (this may take a while)...")
        # model name: distilbert distilled squad; you can change to a different model if you prefer
        _QA_PIPE = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
        logging.info("QA pipeline ready.")
    return _QA_PIPE

# -------------------------
# DB embedding helpers
# -------------------------
def store_embeddings(document_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
    """
    Persist chunk texts and their embeddings into document_embeddings table.
    Embeddings are pickled for storage.
    """
    conn = get_conn()
    cur = conn.cursor()
    # delete existing for doc (if re-upload)
    cur.execute("DELETE FROM document_embeddings WHERE document_id = ?", (document_id,))
    for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        blob = pickle.dumps(emb.astype("float32"))
        cur.execute("INSERT INTO document_embeddings (document_id, chunk_text, embedding, chunk_index) VALUES (?, ?, ?, ?)",
                    (document_id, chunk_text, blob, idx))
    conn.commit()
    conn.close()

def load_embeddings_for_document(document_id: str) -> Tuple[List[str], np.ndarray]:
    """
    Returns list of chunk_texts and numpy array of embeddings (N x D).
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, embedding FROM document_embeddings WHERE document_id = ? ORDER BY chunk_index ASC", (document_id,))
    rows = cur.fetchall()
    conn.close()
    texts = []
    embeddings = []
    for chunk_text, blob in rows:
        try:
            emb = pickle.loads(blob)
            embeddings.append(np.array(emb, dtype="float32"))
            texts.append(chunk_text)
        except Exception as e:
            logging.error(f"load_embeddings_for_document pickling error: {e}")
    if embeddings:
        return texts, np.vstack(embeddings)
    return [], np.empty((0, 0), dtype="float32")

# -------------------------
# Startup event
# -------------------------
@app.on_event("startup")
def startup():
    logging.basicConfig(level=logging.INFO)
    init_database()
    logging.info("Database initialized.")

# -------------------------
# Routes: auth
# -------------------------
@app.post("/auth/signup", response_model=dict)
def signup(user: UserSignup):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (user.username, user.email))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already registered")
    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (user.username, user.email, pwd_context.hash(user.password)))
    conn.commit()
    conn.close()
    return {"message": "User created successfully"}

@app.post("/auth/login", response_model=Token)
def login(user: UserLogin):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (user.username,))
    row = cur.fetchone()
    conn.close()
    if not row or not verify_password(user.password, row[0]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPassword):
    logging.info(f"Password reset requested (placeholder) for {req.email}")
    return {"message": "If the email exists, a reset link has been sent"}

# -------------------------
# Upload endpoint: parse -> chunk -> embed -> persist
# -------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    content = await file.read()

    # size limit 10MB
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 10MB")

    # ensure pdf
    if not validate_pdf_file(content):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_hash = calculate_file_hash(content)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE user_id = ? AND file_hash = ? AND is_archived = 0", (current_user["id"], file_hash))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="This document has already been uploaded")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}.pdf"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(content)

    # extract text
    text = extract_pdf_text(str(stored_path))
    # basic summary (placeholder)
    summary = " ".join(text.split(".")[:3]) if text else "Uploaded. AI insights will be available shortly."

    # chunk text
    chunks = chunk_text(text, max_tokens=400, overlap=50)
    # compute embeddings (if no chunks, make one chunk with empty text)
    embedder = get_embedder()
    if not chunks:
        chunks = [""]  # single empty chunk
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # store metadata + embeddings
    try:
        cur.execute("""
            INSERT INTO documents (id, user_id, filename, original_filename, file_path, size, file_hash, status, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, current_user["id"], stored_name, file.filename or "document.pdf", str(stored_path), len(content), file_hash, "processed", summary))
        conn.commit()
        conn.close()
        # store embeddings separately (pickled)
        store_embeddings(doc_id, chunks, embeddings)
    except Exception as e:
        logging.error(f"upload_document DB error: {e}")
        # cleanup file
        try:
            stored_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Error saving document")

    return {"message": "Document uploaded and processed successfully", "document_id": doc_id, "ai_summary": summary}

# -------------------------
# Documents list
# -------------------------
@app.get("/documents", response_model=List[DocumentResponse])
def get_documents(current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, original_filename, size, upload_date, status, summary FROM documents WHERE user_id = ? AND is_archived = 0 ORDER BY upload_date DESC", (current_user["id"],))
    rows = cur.fetchall()
    conn.close()
    results: List[DocumentResponse] = []
    for r in rows:
        uploaded_at_raw = r[3]
        try:
            uploaded_at = datetime.fromisoformat(uploaded_at_raw)
        except Exception:
            try:
                uploaded_at = datetime.strptime(uploaded_at_raw, "%Y-%m-%d %H:%M:%S")
            except Exception:
                uploaded_at = datetime.now()
        results.append(DocumentResponse(id=r[0], filename=r[1], size=r[2], uploaded_at=uploaded_at, status=r[4] or "processed", summary=r[5]))
    return results

# -------------------------
# View / Download / Archive / Delete
# -------------------------
@app.get("/documents/{doc_id}/download")
def download_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path, original_filename FROM documents WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path, original_filename = row
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path=str(p), filename=original_filename, media_type="application/pdf")

@app.get("/documents/{doc_id}/view")
def view_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM documents WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    p = Path(row[0])
    if not p.exists():
        raise HTTPException(status_code=404, detail="File missing")
    return StreamingResponse(p.open("rb"), media_type="application/pdf", headers={"Content-Disposition": f"inline; filename={p.name}"})

@app.put("/documents/{doc_id}/archive")
def archive_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE documents SET is_archived = 1 WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    if not changed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document archived successfully"}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM documents WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = row[0]
    # remove embeddings and document record
    cur.execute("DELETE FROM document_embeddings WHERE document_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    conn.commit()
    conn.close()
    # remove physical file
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"delete file error: {e}")
    return {"message": "Document deleted successfully"}

# -------------------------
# Q&A endpoint (per-document)
# -------------------------
@app.post("/documents/{doc_id}/ask")
def ask_about_document(doc_id: str, question: Question, current_user: dict = Depends(get_current_user)):
    # 1) verify document
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE id = ? AND user_id = ?", (doc_id, current_user["id"]))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    conn.close()

    # 2) load embeddings & chunks
    texts, embeddings = load_embeddings_for_document(doc_id)
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No content available for Q&A")

    # 3) build FAISS index on the fly and search top-k relevant chunks
    try:
        D, dim = embeddings.shape
    except Exception:
        raise HTTPException(status_code=500, detail="Embeddings not available or corrupted")
    if D == 0:
        raise HTTPException(status_code=400, detail="No embeddings available")

    # Normalize recommended for cosine similarity - but we used L2; we'll use L2 index on float32
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    q_embed = get_embedder().encode([question.question], convert_to_numpy=True).astype("float32")
    k = min(3, embeddings.shape[0])
    distances, indices = index.search(q_embed, k)
    # collect contexts
    contexts = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(texts):
            continue
        contexts.append(texts[idx])
    # merge contexts into a single context (trim if too long)
    context = "\n\n".join(contexts)
    # run QA pipeline
    qa = get_qa_pipeline()
    try:
        # Some QA models expect shorter contexts; if context is too long, truncate
        max_context_len_chars = 20000
        if len(context) > max_context_len_chars:
            context = context[:max_context_len_chars]
        result = qa(question=question.question, context=context)
        # result contains 'answer', 'score', 'start', 'end'
        return {"answer": result.get("answer"), "score": float(result.get("score", 0.0)), "sources": contexts}
    except Exception as e:
        logging.error(f"QA pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Q&A failed")

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
