from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from config import BM25_WEIGHT, FAISS_WEIGHT, RERANK_MIN_SCORE, RERANK_MODEL, RERANK_TOP_K, RETRIEVE_TOP_K


def build_hybrid_retriever(faiss_store, bm25_retriever):
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    bm25_retriever.k = RETRIEVE_TOP_K

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )
    print("Hybrid retriever ready.")
    return retriever


def load_reranker() -> CrossEncoder:
    reranker = CrossEncoder(RERANK_MODEL)
    print("Reranker loaded.")
    return reranker


def _retrieve(hybrid_retriever, query: str):
    if hasattr(hybrid_retriever, "invoke"):
        return hybrid_retriever.invoke(query)
    return hybrid_retriever.get_relevant_documents(query)


def retrieve_and_rerank(query: str, hybrid_retriever, reranker: CrossEncoder):
    docs = _retrieve(hybrid_retriever, query)
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)

    selected = []
    for doc, score in ranked[:RERANK_TOP_K]:
        score_value = float(score)
        if score_value < RERANK_MIN_SCORE:
            continue
        doc.metadata = {**doc.metadata, "rerank_score": score_value}
        selected.append(doc)
    return selected
