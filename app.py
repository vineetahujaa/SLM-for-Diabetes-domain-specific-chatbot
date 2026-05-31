from __future__ import annotations

import json
import os
import re
from pathlib import Path
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Generator, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from config import (
    ALLOWED_HOSTS,
    APP_ENV,
    APP_NAME,
    BASE_DIR,
    CORS_ORIGINS,
    DEBUG,
    FEEDBACK_DB,
    MAX_FEEDBACK_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SUGGESTION_CHARS,
    MODEL_PATH,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    SECURITY_HEADERS_ENABLED,
)
from generator import load_model, stream_answer, stream_direct
from guardrail import (
    MedicalResponsePolicy,
    check_high_risk_medical_input,
    check_unsafe_input,
    enforce_medical_policy,
)
from storage import FeedbackStore, SessionStore

app = FastAPI(title=APP_NAME, version="1.0.0", docs_url="/docs" if DEBUG else None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
validator = MedicalResponsePolicy()
sessions = SessionStore()
feedback_store = FeedbackStore(FEEDBACK_DB)

_llm = None
_hybrid_retriever = None
_reranker = None
_rag_ready = False
_model_lock = threading.Lock()
_rag_lock = threading.Lock()

VALID_MODES = {"guardrail", "rag"}
_SESSION_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.:-]")

_rate_lock = threading.Lock()
_rate_buckets: dict[str, Deque[float]] = defaultdict(deque)

if ALLOWED_HOSTS and ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Session-Id"],
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: str = "guardrail"


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., max_length=MAX_QUESTION_CHARS)
    answer: str = Field(..., max_length=MAX_FEEDBACK_CHARS)
    feedback: str = Field(..., pattern="^(up|down)$")
    suggestion: str = Field(default="", max_length=MAX_SUGGESTION_CHARS)


class SessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    if SECURITY_HEADERS_ENABLED:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
    return response


def _sanitize_session_id(session_id: Optional[str]) -> str:
    cleaned = _SESSION_ID_PATTERN.sub("", (session_id or "default").strip())[:128]
    return cleaned or "default"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _check_rate_limit(request: Request) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    key = _client_ip(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down and try again shortly.",
            )
        bucket.append(now)


def get_llm():
    global _llm
    if _llm is None:
        with _model_lock:
            if _llm is None:
                _llm = load_model()
    return _llm


def _build_rag_indexes():
    from indexing import build_bm25_index, build_faiss_index, write_index_manifest
    from ingestion import chunk_documents, load_pdfs

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    faiss_store = build_faiss_index(chunks)
    bm25 = build_bm25_index(chunks)
    write_index_manifest(chunks_count=len(chunks))
    return faiss_store, bm25


def get_rag_components():
    global _hybrid_retriever, _reranker, _rag_ready

    if _rag_ready:
        return _hybrid_retriever, _reranker

    with _rag_lock:
        if _rag_ready:
            return _hybrid_retriever, _reranker

        from indexing import index_is_current, load_bm25_index, load_faiss_index
        from retrieval import build_hybrid_retriever, load_reranker

        if index_is_current():
            print("Saved RAG index is current. Loading indexes...")
            faiss_store = load_faiss_index()
            bm25 = load_bm25_index()
        else:
            print("RAG index missing or stale. Rebuilding indexes from PDFs...")
            faiss_store, bm25 = _build_rag_indexes()

        _hybrid_retriever = build_hybrid_retriever(faiss_store, bm25)
        _reranker = load_reranker()
        _rag_ready = True
        return _hybrid_retriever, _reranker


def sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


_META_PATTERNS = re.compile(
    r"(what (question|did i|have i)|which question|questions (i|i've)|"
    r"what (did|have) (i|we) (ask|discuss|talk|cover|say)|"
    r"summarize (our|the|this) (chat|conversation|session|discussion)|"
    r"what (we|have we) (talked|discussed|covered)|"
    r"list (my|the|all) question|previous question|history of (our|this))",
    re.IGNORECASE,
)


def _is_meta_question(question: str) -> bool:
    return bool(_META_PATTERNS.search(question))


def _meta_response(history: list) -> str:
    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    if not user_turns:
        return "We haven't discussed anything yet. Feel free to ask me a diabetes-related question!"
    listed = "\n".join(f"{i+1}. {q}" for i, q in enumerate(user_turns))
    note = (
        f"\n\n*(I can see the last {len(user_turns)} question{'s' if len(user_turns) != 1 else ''} "
        f"from our current session.)*"
    )
    return f"Here are the questions you've asked so far:\n\n{listed}{note}"


def _safe_error(exc: Exception) -> str:
    if DEBUG or APP_ENV != "production":
        return f"Something went wrong: {exc}"
    return "Something went wrong while processing the request. Please try again."


def _collect_answer(token_iter, t0: float) -> tuple[str, int, float]:
    full_answer = ""
    token_count = 0
    ttft = 0.0
    for token in token_iter:
        if token_count == 0:
            ttft = round(time.perf_counter() - t0, 3)
        full_answer += token
        token_count += 1
    return full_answer, token_count, ttft


def _stream_and_collect(token_iter, t0: float):
    """Stream tokens live as SSE while collecting the full answer.
    Yields SSE strings, then returns (full_answer, token_count, ttft) via StopIteration.
    """
    full_answer = ""
    token_count = 0
    ttft = 0.0
    buf: list[str] = []
    last_flush = time.perf_counter()

    def flush(force: bool = False):
        nonlocal last_flush
        if not buf:
            return
        now = time.perf_counter()
        if force or (now - last_flush) >= 0.04 or len(buf) >= 16:
            chunk = "".join(buf)
            buf.clear()
            last_flush = now
            yield sse({"type": "token", "v": chunk})

    for token in token_iter:
        if token_count == 0:
            ttft = round(time.perf_counter() - t0, 3)
            yield sse({"type": "ttft", "v": ttft})
        full_answer += token
        token_count += 1
        buf.append(token)
        yield from flush()
    yield from flush(force=True)
    return full_answer, token_count, ttft


def _sources_markdown(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    seen = set()
    lines = ["\n\n---\n**Sources:**"]
    for src in sources:
        key = (src.get("source"), src.get("page"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {src.get('source', 'unknown')} | Page {src.get('page', '?')}")
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "env": APP_ENV, "sessions": sessions.count()}


@app.get("/ready")
async def ready():
    rag_index_current = False
    rag_error = None
    try:
        from indexing import index_is_current

        rag_index_current = index_is_current()
    except Exception as exc:  # keep readiness endpoint useful even before RAG deps are installed
        rag_error = str(exc) if DEBUG else exc.__class__.__name__

    ready_state = MODEL_PATH.exists() and feedback_store.healthy() and rag_error is None
    return {
        "status": "ready" if ready_state else "degraded",
        "model_present": MODEL_PATH.exists(),
        "feedback_db": feedback_store.healthy(),
        "rag_index_current": rag_index_current,
        "rag_error": rag_error,
    }


@app.post("/chat_stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    x_session_id: Optional[str] = Header(default="default"),
):
    question = (body.question or "").strip()
    mode = body.mode if body.mode in VALID_MODES else "guardrail"
    session_id = _sanitize_session_id(x_session_id)
    _check_rate_limit(request)

    session = sessions.get(session_id)
    history = session.get("history", [])
    prev_q = session.get("last_question")
    prev_r = session.get("last_response")
    prev_p = session.get("last_passed", False)

    def generate() -> Generator[str, None, None]:
        t0 = time.perf_counter()
        full_answer = ""
        token_count = 0
        ttft = 0.0
        is_valid = True
        source_text = ""
        sources = []

        try:
            safety = check_unsafe_input(question)
            if not safety.get("safe", True):
                yield sse({"type": "replace", "v": safety.get("reason", "Request blocked."), "passed": False})
                yield sse({"type": "status", "v": "done"})
                return

            if _is_meta_question(question):
                msg = _meta_response(history)
                yield sse({"type": "token", "v": msg})
                yield sse({"type": "valid", "passed": True})
                yield sse({"type": "status", "v": "done"})
                sessions.update(session_id, question, msg, True)
                return

            urgent = check_high_risk_medical_input(question)
            if urgent.get("urgent"):
                yield sse({"type": "replace", "v": urgent.get("reason", "Please seek urgent medical care."), "passed": False})
                yield sse({"type": "status", "v": "done"})
                sessions.update(session_id, question, urgent.get("reason", ""), False)
                return

            llm = get_llm()

            if mode == "rag":
                yield sse({"type": "searching"})
                from retrieval import retrieve_and_rerank

                hybrid_retriever, reranker = get_rag_components()
                top_docs = retrieve_and_rerank(question, hybrid_retriever, reranker)
                if not top_docs:
                    msg = (
                        "I couldn't find relevant document context for that question in the indexed PDFs. "
                        "Please add the right PDF to the data folder and rebuild the index, or ask a general diabetes question."
                    )
                    yield sse({"type": "replace", "v": msg, "passed": False})
                    yield sse({"type": "status", "v": "done"})
                    sessions.update(session_id, question, msg, False)
                    return

                sources = [
                    {
                        "page": doc.metadata.get("page"),
                        "source": doc.metadata.get("source"),
                        "score": doc.metadata.get("rerank_score"),
                    }
                    for doc in top_docs
                ]
                yield sse({"type": "generating"})
                token_iter = stream_answer(question, top_docs, llm, history=history)
            else:
                token_iter = stream_direct(question, llm, history=history)

            # Stream tokens live, collect full answer simultaneously
            streamer = _stream_and_collect(token_iter, t0)
            try:
                while True:
                    yield next(streamer)
            except StopIteration as stop:
                full_answer, token_count, ttft = stop.value or ("", 0, 0.0)

            # Validate after generation — if fail, replace streamed tokens with refusal
            is_valid, final_answer, reason = enforce_medical_policy(
                full_answer,
                question,
                validator=validator,
                previous_question=prev_q,
                previous_response=prev_r,
                previous_passed=prev_p,
                llm=llm,
            )
            full_answer = final_answer

            if not is_valid:
                print(f"Guardrail blocked response: {reason}")
                yield sse({"type": "replace", "v": final_answer, "passed": False})
            else:
                if mode == "rag":
                    source_text = _sources_markdown(sources)
                    if source_text:
                        yield sse({"type": "sources", "v": source_text})
                yield sse({"type": "valid", "passed": True})

        except Exception as exc:
            print(f"Chat stream error: {exc}")
            yield sse({"type": "replace", "v": _safe_error(exc), "passed": False})
            yield sse({"type": "status", "v": "done"})
            return

        total = time.perf_counter() - t0
        tps = round(token_count / max(total, 0.1), 1)
        yield sse({"type": "tps", "v": tps})
        yield sse({"type": "status", "v": "done"})

        if full_answer:
            sessions.update(session_id, question, full_answer + source_text, bool(is_valid))

        print(
            f"[{mode.upper()}] TTFT={ttft}s TPS={tps} tokens={token_count} "
            f"valid={is_valid} q={question[:60]!r}"
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request):
    feedback_store.add(
        {
            "session_id": _sanitize_session_id(body.session_id),
            "question": body.question.strip(),
            "answer": body.answer.strip(),
            "feedback": body.feedback,
            "suggestion": body.suggestion.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"client_host": request.client.host if request.client else None},
        }
    )
    return JSONResponse({"status": "ok"})


@app.post("/api/clear_memory")
async def clear_memory(x_session_id: Optional[str] = Header(default="default")):
    sessions.clear(_sanitize_session_id(x_session_id))
    return JSONResponse({"status": "cleared"})


@app.post("/end_session")
async def end_session(body: SessionRequest):
    sessions.end(_sanitize_session_id(body.session_id))
    return JSONResponse({"status": "ended"})


@app.post("/api/rebuild_index")
async def rebuild_index(request: Request):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if admin_token:
        supplied = request.headers.get("x-admin-token", "")
        if supplied != admin_token:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    elif APP_ENV == "production":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ADMIN_TOKEN is required in production")

    global _hybrid_retriever, _reranker, _rag_ready
    with _rag_lock:
        from retrieval import build_hybrid_retriever, load_reranker

        faiss_store, bm25 = _build_rag_indexes()
        _hybrid_retriever = build_hybrid_retriever(faiss_store, bm25)
        _reranker = load_reranker()
        _rag_ready = True
    return JSONResponse({"status": "rebuilt"})


@app.get("/api/documents")
async def list_documents():
    from config import DATA_DIR
    files = [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(DATA_DIR.rglob("*.pdf"))
    ]
    return JSONResponse({"documents": files})


@app.post("/api/upload_document")
async def upload_document(file: UploadFile = File(...)):
    from config import DATA_DIR
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename
    dest.write_bytes(content)

    global _rag_ready
    _rag_ready = False
    return JSONResponse({"status": "uploaded", "filename": filename})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=DEBUG)
