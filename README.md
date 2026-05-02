# Chat with Marine Biology Docs

A portfolio-quality FastAPI RAG app that answers marine biology questions from textbook PDF and text excerpts. It ingests local source documents, chunks and embeds them with LlamaIndex, stores vectors in persistent Chroma storage, reranks retrieved evidence, and returns answers with citations from actual retrieved chunk metadata.

## Why This Project Exists

This app is an experiment in building practical retrieval-augmented generation pipelines rather than a plain chatbot. The goal is to explore how LlamaIndex can coordinate document loading, chunking, embeddings, vector storage, retrieval, reranking, and prompt assembly while keeping the application code understandable.

Marine biology chapters are a good test domain because answers need grounding in specific source passages. The app therefore prioritizes citation reliability: retrieved chunks carry filename, page, and chunk metadata, and the API returns those citations directly from the database instead of asking the language model to invent them. This makes the project useful for experimenting with RAG quality questions such as chunk size, top-k retrieval, reranker behavior, and whether the final answer is actually supported by the retrieved context.

## What The App Does

- Loads `.pdf` and `.txt` files from `data/raw_docs/`.
- Extracts PDF text page-by-page so citations can include page numbers.
- Chunks documents with LlamaIndex and stores `filename`, `page`, `chunk_id`, and optional `section` metadata.
- Persists embeddings in Chroma under `data/storage/chroma/`.
- Retrieves the top 20 candidate chunks, reranks them with a sentence-transformers cross-encoder, and sends the top 5 chunks to the LLM.
- Returns an answer plus a `sources` array built from retrieved metadata, not invented by the model.

## Project Structure

```text
app/
  main.py       # FastAPI routes
  ingest.py     # document loading, chunking, embedding, persistent indexing
  rag.py        # retrieval, reranking, prompt construction, cited answers
  schemas.py    # Pydantic request/response models
  config.py     # .env-backed settings
  utils.py      # shared helpers
data/
  raw_docs/     # add 10-20 marine biology PDFs or text files here
  storage/      # persistent Chroma vector database
requirements.txt
README.md
.env.example
```

## How RAG Works Here

1. Ingestion reads source files from `data/raw_docs/`.
2. PDF pages and text files become LlamaIndex `Document` objects with metadata.
3. `SentenceSplitter` creates overlapping chunks.
4. Each chunk receives a stable `chunk_id`.
5. The embedding model converts chunks into vectors stored in Chroma.
6. `/ask` embeds the question and retrieves the top 20 vector matches.
7. A cross-encoder reranker scores question/chunk pairs.
8. The top 5 chunks are placed into a strict prompt that tells the LLM to answer only from the excerpts.
9. The API response includes citations from the selected chunk metadata.

## Setup

Python 3.11 or 3.12 is recommended because vector database and local ML packages can lag behind brand-new Python releases.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set at least one LLM provider key:

```env
MARINE_RAG_LLM_PROVIDER=openai
MARINE_RAG_OPENAI_API_KEY=your_openai_key
MARINE_RAG_OPENAI_MODEL=gpt-4o-mini
MARINE_RAG_EMBEDDING_PROVIDER=openai
MARINE_RAG_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For Anthropic answers with OpenAI embeddings:

```env
MARINE_RAG_LLM_PROVIDER=anthropic
MARINE_RAG_ANTHROPIC_API_KEY=your_anthropic_key
MARINE_RAG_OPENAI_API_KEY=your_openai_key_for_embeddings
```

For local embeddings, set:

```env
MARINE_RAG_EMBEDDING_PROVIDER=local
MARINE_RAG_LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

## Add Documents

Place 10-20 marine biology textbook chapter PDFs or `.txt` files in:

```text
data/raw_docs/
```

PDF citations include page numbers. Text files return `null` for page because there is no page structure.

## Run Ingestion

From the repo root:

```powershell
python -m app.ingest
```

Or start the API and trigger ingestion:

```powershell
curl -X POST http://127.0.0.1:8000/ingest
```

## Start FastAPI

```powershell
uvicorn app.main:app --reload
```

Open the API docs at `http://127.0.0.1:8000/docs`.

## API

Health check:

```powershell
curl http://127.0.0.1:8000/
```

Ask a question:

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What role do coral reefs play in marine ecosystems?"}'
```

Example response:

```json
{
  "answer": "Coral reefs support marine biodiversity by providing habitat, food sources, and breeding grounds for many species (marine_biology_chapter_3.pdf, p. 12).",
  "sources": [
    {
      "filename": "marine_biology_chapter_3.pdf",
      "page": 12,
      "chunk_id": "chunk_0034",
      "excerpt": "Coral reefs provide habitat and shelter for many marine organisms..."
    }
  ]
}
```

If the retrieved context does not support an answer, the model is instructed to respond:

```text
I don’t know based on the provided documents.
```

## Configuration

Key environment variables:

- `MARINE_RAG_OPENAI_API_KEY`: OpenAI key for OpenAI LLMs or embeddings.
- `MARINE_RAG_ANTHROPIC_API_KEY`: Anthropic key if `MARINE_RAG_LLM_PROVIDER=anthropic`.
- `MARINE_RAG_LLM_PROVIDER`: `openai` or `anthropic`.
- `MARINE_RAG_EMBEDDING_PROVIDER`: `openai` or `local`.
- `MARINE_RAG_SIMILARITY_TOP_K`: candidate retrieval count, default `20`.
- `MARINE_RAG_FINAL_TOP_K`: final chunks sent to the LLM, default `5`.
- `MARINE_RAG_ENABLE_RERANKING`: enables cross-encoder reranking.

## Limitations

- PDF extraction uses `pypdf`, which can struggle with scanned pages, complex tables, and multi-column layouts.
- The default vector search is semantic only, so exact keyword matching may miss useful chunks.
- The reranker improves relevance but adds latency and may download a local model on first use.
- The API returns the chunks used as evidence, but it does not highlight exact answer spans inside PDFs yet.

## Future Improvements

- Better PDF parsing with layout-aware extraction or OCR for scanned chapters.
- Hybrid search combining vector similarity with BM25 keyword retrieval.
- Better reranking with Cohere Rerank or a domain-tuned cross-encoder.
- UI frontend for browsing answers and citations.
- User uploads with validation and background ingestion jobs.
- Evaluation metrics for retrieval quality, citation faithfulness, and answer correctness.
- Citation highlighting that links answers back to exact PDF spans.
