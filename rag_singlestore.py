from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_singlestore.vectorstores import SingleStoreVectorStore


@dataclass
class RagConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    table_name: str = "insightmate_vectors"


def chunk_documents(docs: List[Document], cfg: RagConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(docs)


def _get_table_name() -> str:
    return os.getenv("SINGLESTORE_TABLE", "insightmate_vectors")


def get_vectorstore(table_name: str | None = None) -> SingleStoreVectorStore:
    """
    Returns a SingleStore-backed vector store.
    The integration package handles table creation/usage internally.
    """
    embeddings = OpenAIEmbeddings()

    return SingleStoreVectorStore(
        embedding=embeddings,
        host=os.getenv("SINGLESTORE_HOST"),
        port=int(os.getenv("SINGLESTORE_PORT", "3306")),
        user=os.getenv("SINGLESTORE_USER"),
        password=os.getenv("SINGLESTORE_PASSWORD"),
        database=os.getenv("SINGLESTORE_DATABASE"),
        table_name=table_name or _get_table_name(),
    )


def upsert_documents(vs: SingleStoreVectorStore, chunks: List[Document]) -> None:
    """
    Adds documents to SingleStore (persists across app restarts).
    """
    vs.add_documents(chunks)


def get_retriever(vs: SingleStoreVectorStore, top_k: int):
    return vs.as_retriever(search_kwargs={"k": top_k})


def format_sources(docs: List[Document], max_chars: int = 900) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("source", f"Source {i}")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        out.append((title, snippet))
    return out
