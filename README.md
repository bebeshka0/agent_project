## Project Structure

```
agent_project/
├── app.py                 # Main Streamlit application with LLM agent
├── ingest_documents.py     # Document indexing script for RAG system
├── docker-compose.yaml     # PostgreSQL with pgvector container configuration
├── documents/              # Folder for PDF articles (to be indexed)
└── venv/                  # Python virtual environment
```

### `app.py`

**Purpose:** Main Streamlit web application providing chat interface for the first LLM agent.

**Responsibilities:**
- Creates interactive chat interface using Streamlit
- Initializes and manages LangChain chain with Ollama LLM
- Handles user queries and displays responses
- Maintains chat history in session state

**Technical Details:**
- **LLM Model:** `phi3:mini` (via Ollama)
- **Framework:** LangChain with chain composition pattern
- **Components:**
  - `ChatOllama`: Connects to locally deployed Ollama service
  - `ChatPromptTemplate`: System prompt configured for ML education
  - `StrOutputParser`: Converts LLM output to string format
  - `RunnablePassthrough`: Passes user input through the chain
- **System Prompt:** Configured as expert AI tutor specializing in machine learning education
- **Session State:** Stores chain instance and message history to avoid reinitialization

**Key Functions:**
- `initialize_chain()`: Sets up LangChain chain with prompt template and LLM
- `main()`: Streamlit application entry point

### `ingest_documents.py`

**Purpose:** Document ingestion utility for preparing RAG system. Processes PDF files and stores them as vector embeddings in PostgreSQL.

**Responsibilities:**
- Loads PDF documents from `documents/` folder
- Splits documents into chunks for better retrieval
- Generates embeddings using HuggingFace model
- Stores vectors in PostgreSQL with pgvector extension

**Technical Details:**
- **Document Loader:** `PyPDFLoader` from `langchain_community`
- **Text Splitter:** `RecursiveCharacterTextSplitter` with:
  - `chunk_size`: 1000 characters
  - `chunk_overlap`: 200 characters (for context preservation)
- **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace)
  - Lightweight model (~80MB)
  - Generates 384-dimensional vectors
- **Vector Store:** `PGVector` from `langchain-postgres`
- **Database:** PostgreSQL with pgvector extension
- **Collection Name:** `ml_articles_collection`

**Key Functions:**
- `load_pdf_documents(folder_path)`: Loads all PDF files from specified folder
- `split_documents(documents, chunk_size, chunk_overlap)`: Splits documents into chunks
- `ingest_documents()`: Main ingestion pipeline

**Configuration Constants:**
- `DOCS_FOLDER`: "documents" - folder containing PDF files
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" - HuggingFace embedding model
- `CONNECTION_STRING`: PostgreSQL connection string
- `COLLECTION_NAME`: "ml_articles_collection" - vector collection name


### Vector Database
- Uses pgvector extension for efficient similarity search
- Embeddings stored as `vector` type in PostgreSQL
- Collection-based organization for different document sets

### Text Chunking Strategy
- Recursive character splitting preserves document structure
- Overlap between chunks maintains context
- Chunk size optimized for embedding model context window

### LangChain Chain Pattern
- Uses LangChain's chain composition with pipe operator (`|`)
- Enables easy extension and modification
- RunnablePassthrough allows flexible input handling

## Future Enhancements

- [ ] Implement second RAG agent with vector database retrieval
- [ ] Add agent routing/selection mechanism
- [ ] Implement conversation memory for RAG agent
- [ ] Add document metadata filtering
