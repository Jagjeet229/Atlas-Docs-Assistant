from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
import singlestoredb as s2

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from loaders import uploaded_files_to_documents
from prompts import SYSTEM_PROMPT
from rag_singlestore import RagConfig, chunk_documents, format_sources, get_vectorstore

load_dotenv()

st.set_page_config(page_title="AtlasDocs — Vector Assistant", page_icon="🗂️", layout="wide")


# ---------------- Helpers ----------------
def drop_table(table_name: str):
    conn = s2.connect(
        host=os.getenv("SINGLESTORE_HOST"),
        port=int(os.getenv("SINGLESTORE_PORT", "3306")),
        user=os.getenv("SINGLESTORE_USER"),
        password=os.getenv("SINGLESTORE_PASSWORD"),
        database=os.getenv("SINGLESTORE_DATABASE"),
    )
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
    conn.commit()
    conn.close()


# ---------------- Sidebar ----------------
# Load configuration from environment (no sidebar customization)
api_key = os.getenv("OPENAI_API_KEY", "")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
top_k = int(os.getenv("TOP_K", "8"))
table_name = os.getenv("SINGLESTORE_TABLE", "docchat_vectors")
max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "50"))

uploaded = st.sidebar.file_uploader(
    "Upload documents (TXT/MD/PDF)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

# Validate uploaded file sizes and stop if any file is too large
if uploaded:
    oversized = [f for f in uploaded if getattr(f, "size", 0) > max_upload_mb * 1024 * 1024]
    if oversized:
        names = ", ".join(f.name for f in oversized)
        st.sidebar.error(f"File(s) too large: {names}. Maximum per-file size is {max_upload_mb} MB.")
        st.stop()

col1, col2 = st.sidebar.columns(2)
build_index = col1.button("⚙️ Index Documents", use_container_width=True, disabled=not uploaded)
clear_chat = col2.button("🧹 Clear Conversation", use_container_width=True)

if clear_chat:
    st.session_state.pop("messages", None)


# ---------------- Main UI ----------------
st.markdown(
    "### 🗂️ AtlasDocs — Vector Knowledge — *Upload documents, index them into SingleStore, and ask questions with cited sources and context-aware answers.*"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for title, snippet in m["sources"]:
                    st.markdown(f"**{title}**")
                    st.write(snippet)


# Build/Upsert
if build_index:
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        st.stop()

    cfg = RagConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        table_name=table_name,
    )

    with st.spinner("Reading uploaded files…"):
        docs = uploaded_files_to_documents(uploaded)

    if not docs:
        st.warning("No readable text found in uploaded files.")
        st.stop()

    with st.spinner("Chunking documents…"):
        chunks = chunk_documents(docs, cfg)

    # Clear existing embeddings table to avoid old vectors (do this just before indexing)
    with st.spinner(f"Clearing existing embeddings table `{table_name}`..."):
        drop_table(table_name)

    with st.spinner("Connecting to SingleStore…"):
        vs = get_vectorstore(table_name=table_name)

    st.info(f"Indexing {len(chunks)} chunks (large PDFs can take time)…")
    progress = st.progress(0)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vs.add_documents(chunks[i : i + batch_size])
        progress.progress(min((i + batch_size) / len(chunks), 1.0))

    st.session_state["vectorstore"] = vs
    st.success(f"Index complete ✅ Files: {len(docs)} | Chunks: {len(chunks)} | Table: {table_name}")


# Chat
user_q = st.chat_input("Ask a question about your uploaded documents…")

if user_q:
    if "vectorstore" not in st.session_state:
        st.warning("Upload documents and click **Index Documents** first.")
        st.stop()

    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    llm = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=True,
    )

    vs = st.session_state["vectorstore"]

    # Retriever
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    # Retrieve docs
    docs = retriever.invoke(user_q)

    # 🔎 Debug retrieved chunks
    with st.expander("🔎 Retrieved chunks (debug)"):
        st.write(f"Retrieved: {len(docs)} chunks")
        for i, d in enumerate(docs, 1):
            st.markdown(f"### Chunk {i} — {d.metadata.get('source', '')}")
            st.write(d.page_content[:1500])

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        messages = prompt.format_messages(context=context, question=user_q)

        for chunk in llm.stream(messages):
            token = getattr(chunk, "content", "") or ""
            answer += token
            placeholder.markdown(answer)

        sources = format_sources(docs)
        if sources:
            with st.expander("Sources"):
                for title, snippet in sources:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
