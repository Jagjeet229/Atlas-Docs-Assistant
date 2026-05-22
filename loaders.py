from __future__ import annotations

import io
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def uploaded_files_to_documents(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        data = f.read()
        name = f.name

        if name.lower().endswith(".pdf"):
            text = _read_pdf(data)
        else:
            text = _read_text(data)

        text = (text or "").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": name},
            )
        )
    return docs
