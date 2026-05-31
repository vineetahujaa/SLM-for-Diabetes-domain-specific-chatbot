from pathlib import Path

import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR


def load_pdfs(pdf_folder: str | Path = DATA_DIR) -> list[Document]:
    folder = Path(pdf_folder)
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder}")

    documents: list[Document] = []
    for pdf_path in pdf_files:
        added_pages = 0
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                text = doc[page_num].get_text().strip()
                if len(text) < 50:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num + 1},
                    )
                )
                added_pages += 1
        finally:
            doc.close()
        print(f"Loaded {pdf_path.name}: {added_pages} text pages")

    print(f"Total pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
    print(f"Total chunks: {len(chunks)}")
    return chunks
