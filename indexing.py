from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    ALLOW_DANGEROUS_INDEX_DESERIALIZATION,
    BM25_PATH,
    DATA_DIR,
    EMBED_MODEL,
    FAISS_DIR,
    INDEX_MANIFEST_PATH,
    RETRIEVE_TOP_K,
)


def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest(pdf_folder: Path = DATA_DIR) -> dict[str, Any]:
    files = []
    for path in sorted(Path(pdf_folder).rglob("*.pdf")):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "version": 1,
        "data_dir": str(Path(pdf_folder).resolve()),
        "files": files,
        "source_hash": combined,
    }


def build_faiss_index(chunks):
    if not chunks:
        raise ValueError("Cannot build FAISS index: no chunks provided.")

    embedder = get_embedder()
    faiss_store = FAISS.from_documents(chunks, embedder)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(FAISS_DIR))
    print("FAISS index saved.")
    return faiss_store


def load_faiss_index():
    if not ALLOW_DANGEROUS_INDEX_DESERIALIZATION:
        raise RuntimeError(
            "FAISS index loading requires pickle deserialization. Set "
            "ALLOW_DANGEROUS_INDEX_DESERIALIZATION=true only for trusted local indexes, "
            "or rebuild indexes instead of loading saved files."
        )
    embedder = get_embedder()
    faiss_store = FAISS.load_local(
        str(FAISS_DIR),
        embedder,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded.")
    return faiss_store


def build_bm25_index(chunks):
    if not chunks:
        raise ValueError("Cannot build BM25 index: no chunks provided.")

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = RETRIEVE_TOP_K
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25, f)
    print("BM25 index saved.")
    return bm25


def load_bm25_index():
    if not ALLOW_DANGEROUS_INDEX_DESERIALIZATION:
        raise RuntimeError(
            "BM25 index loading requires pickle deserialization. Set "
            "ALLOW_DANGEROUS_INDEX_DESERIALIZATION=true only for trusted local indexes."
        )
    with BM25_PATH.open("rb") as f:
        bm25 = pickle.load(f)
    bm25.k = RETRIEVE_TOP_K
    print("BM25 index loaded.")
    return bm25


def write_index_manifest(*, chunks_count: int, pdf_folder: Path = DATA_DIR) -> dict[str, Any]:
    manifest = source_manifest(pdf_folder)
    manifest.update(
        {
            "chunks_count": chunks_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "faiss_index": str(FAISS_DIR),
            "bm25_path": str(BM25_PATH),
        }
    )
    INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("Index manifest saved.")
    return manifest


def read_index_manifest() -> dict[str, Any] | None:
    if not INDEX_MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def index_files_exist() -> bool:
    return (
        (FAISS_DIR / "index.faiss").exists()
        and (FAISS_DIR / "index.pkl").exists()
        and BM25_PATH.exists()
    )


def index_is_current(pdf_folder: Path = DATA_DIR) -> bool:
    if not index_files_exist():
        return False
    saved = read_index_manifest()
    if not saved:
        return False
    current = source_manifest(pdf_folder)
    return saved.get("source_hash") == current.get("source_hash")


