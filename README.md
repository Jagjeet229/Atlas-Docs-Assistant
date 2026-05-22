## AtlasDocs Copy (project2)

This folder contains a standalone copy of the AtlasDocs document QA app.

### What it does
- `app.py` — AtlasDocs
  - Upload documents (PDF, TXT, or Markdown)
  - Chunk and embed document text
  - Store embeddings in SingleStore
  - Query your documents with answers that include source citations

### Files
- `app.py` — Streamlit app entry point
- `loaders.py` — PDF / text file loading and conversion to LangChain documents
- `prompts.py` — system prompt used for the assistant
- `rag_singlestore.py` — chunking, embeddings, and SingleStore vector store helpers
- `requirements.txt` — Python dependencies for this app
- `.env` — environment configuration for OpenAI and SingleStore

### Usage
1. Install requirements:
```bash
pip install -r project2/requirements.txt
```
2. Configure `project2/.env` with your OpenAI and SingleStore values.
3. Run the app:
```bash
streamlit run project2/app.py
```
