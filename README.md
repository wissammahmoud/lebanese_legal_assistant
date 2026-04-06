# ADL Legal Assistant

An AI-powered legal assistant specializing in Lebanese law, built for lawyers, law students, and legal professionals. It provides accurate legal information based on Lebanese statutes, regulations, and court rulings using a RAG (Retrieval-Augmented Generation) pipeline.

## Features

- Arabic-first legal Q&A grounded in Lebanese law
- Vector search over 8,000+ legal articles and rulings (Milvus / Zilliz Cloud)
- Intent classification — greetings and off-topic questions are filtered before hitting the pipeline
- Query rewriting for improved retrieval accuracy
- Streaming responses with source citations
- Follow-up question suggestions per answer
- Online evaluation via OpenEvals + LangSmith tracing

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python) |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim) |
| Vector DB | Zilliz Cloud (Milvus-compatible) |
| Caching | Redis |
| Tracing | LangSmith |
| Deployment | Vercel |

## Project Structure

```
app/
  services/
    rag_service.py          # Core RAG pipeline
    vector_store_service.py # Milvus/Zilliz search
    embedding_service.py    # OpenAI embeddings
    llm_service.py          # LLM generation & streaming
    query_rewriter_service.py
    drafting_service.py     # Document drafting templates
data_pipeline/
  ingest_data.py            # Ingest Excel files → Milvus
  milvus_setup.py           # Create Milvus/Zilliz collection schema
  migrate_metadata.py       # Fix metadata in existing Milvus records
  transfer_to_zilliz.py     # Transfer local Milvus → Zilliz Cloud
frontend/                   # Frontend assets
```

## Environment Variables

Create a `.env` file at the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Zilliz Cloud (Milvus)
MILVUS_URI=https://your-cluster.zillizcloud.com
MILVUS_TOKEN=your_api_key
MILVUS_COLLECTION_NAME=lebanese_laws
MILVUS_DIMENSION=1536

# Redis
REDIS_URL=redis://localhost:6379/0

# LangSmith (optional)
LANGCHAIN_TRACING_V2=True
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=lebanese-legal-assistant
```

## Data Pipeline

### 1. Create the collection
```bash
python data_pipeline/milvus_setup.py
```

### 2. Ingest Excel files
Excel files must have columns: `text_content`, `source_type`, and optionally `Metadata` (any casing).
```bash
python data_pipeline/ingest_data.py data/
```

### 3. Migrate metadata (if ingested with wrong column casing)
```bash
python data_pipeline/migrate_metadata.py data/
```

### 4. Transfer local Milvus → Zilliz Cloud
```bash
python data_pipeline/transfer_to_zilliz.py \
  --zilliz-uri  https://your-cluster.zillizcloud.com \
  --zilliz-token your_api_key
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
