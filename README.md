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
