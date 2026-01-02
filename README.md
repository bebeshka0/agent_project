## Project Structure

```
agent_project/
├── app.py                 # Main Streamlit application (UI & Logic)
├── ingest_documents.py    # Document indexing script for RAG system
├── router_agent.py        # Semantic router to classify user queries
├── rag_agent.py           # RAG agent with FlashRank reranking
├── rag_vector_store.py    # Vector store connection logic
├── documents/             # Default folder for PDF articles
```

## Key Features

- **Multi-Agent Architecture**:
  - **Router Agent**: Decides if a question is theoretical (sent to Tutor) or context-specific (sent to RAG).
  - **Tutor Agent**: Explains general ML concepts using a large language model (xAI Grok).
  - **RAG Agent**: Retrieves answers from your local PDF documents with high precision.

- **Advanced RAG Pipeline**:
  - **Semantic Chunking**: Splits text based on meaning rather than fixed character counts.
  - **Reranking**: Uses `FlashRank` (Lightweight/Fast) to re-order retrieved documents for better relevance.
  - **Vector Search**: Uses PostgreSQL with `pgvector` for efficient similarity search.

- **Interactive UI**:
  - **Chat Interface**: Clean, WhatsApp-like chat experience.
  - **Document Upload**: Drag-and-drop PDF upload directly in the sidebar to instantly expand the agent's knowledge base.

## Configuration (recommended)

This project is configured via environment variables. For Docker Compose deployments, keep secrets in a local `.env` file on the target machine.

1. Create your local `.env` from the template:

```bash
cp env.example .env
```

2. Edit `.env` and set at least:
   - `POSTGRES_CONNECTION_STRING` (or `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`)
   - `XAI_API_KEY`

3. Start the stack:

```bash
docker compose -f docker-compose.app.yaml up -d --build
```

Notes:
- The `.env` file is intentionally ignored by git; each deployer should maintain their own copy.
- On first start, the database tables used by `langchain_postgres` are created automatically by the application when needed.
