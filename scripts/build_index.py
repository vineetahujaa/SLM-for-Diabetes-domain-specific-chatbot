from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion import chunk_documents, load_pdfs
from indexing import build_bm25_index, build_faiss_index, write_index_manifest


def main() -> None:
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    build_faiss_index(chunks)
    build_bm25_index(chunks)
    write_index_manifest(chunks_count=len(chunks))
    print(f"Built index for {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
