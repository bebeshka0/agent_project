import os
from typing import Optional

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

load_dotenv()

DOCS_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ml_articles_collection")
CONNECTION_STRING: str = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql://postgres:mysecretpassword@localhost:5432/vector_db",
)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def initialize_vector_store() -> Optional[PGVector]:
    try:
        embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store: PGVector = PGVector(
            embeddings=embeddings,
            collection_name=DOCS_COLLECTION_NAME,
            connection=CONNECTION_STRING,
        )

        return vector_store
    except Exception as exc:

        print("Failed to initialize PGVector vector store.")
        print(f"Details: {exc}")
        return None


def get_retriever(k: int = 5) -> Optional[VectorStoreRetriever]:
    vector_store: Optional[PGVector] = initialize_vector_store()

    if vector_store is None:
        return None

    retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever


def get_pgvector_index_sql(table_name: str, column_name: str) -> str:

    return (
        "CREATE INDEX IF NOT EXISTS "
        f"idx_{table_name}_{column_name}_cosine_hnsw "
        f"ON {table_name} USING hnsw ({column_name} vector_cosine_ops);"
    )



