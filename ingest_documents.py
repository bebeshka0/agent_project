import os
import re
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

load_dotenv()

DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "documents")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-base",
)
CONNECTION_STRING: str = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql://postgres:mysecretpassword@localhost:5432/vector_db",
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ml_articles_collection")


def _normalize_for_hash(text: str) -> str:

    normalized: str = clean_text_content(text)
    normalized = normalized.lower()
    normalized = normalized.replace("ё", "е")
    return normalized


def _compute_text_hash(pages: List[Document]) -> str:

    full_text: str = "\n".join(page.page_content for page in pages)
    normalized: str = _normalize_for_hash(full_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _make_doc_id(original_filename: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, original_filename).hex


def _attach_document_metadata(
    pages: List[Document],
    *,
    doc_id: str,
    original_filename: str,
    text_hash: str,
    version: int,
    ingested_at: str,
) -> None:

    for page in pages:
        page.metadata.update(
            {
                "doc_id": doc_id,
                "original_filename": original_filename,
                "text_hash": text_hash,
                "version": version,
                "ingested_at": ingested_at,
            }
        )


def _get_engine() -> Engine:

    return create_engine(CONNECTION_STRING)


def _get_collection_id(engine: Engine) -> Optional[str]:

    query = text("SELECT uuid FROM langchain_pg_collection WHERE name = :name LIMIT 1;")
    with engine.connect() as conn:
        row = conn.execute(query, {"name": COLLECTION_NAME}).fetchone()
    return str(row[0]) if row is not None else None


def _document_exists_by_text_hash(engine: Engine, *, collection_id: str, text_hash: str) -> bool:
    query = text(
        """
        SELECT 1
        FROM langchain_pg_embedding
        WHERE collection_id = :collection_id
          AND cmetadata ->> 'text_hash' = :text_hash
        LIMIT 1;
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"collection_id": collection_id, "text_hash": text_hash}).fetchone()
    return row is not None


def _group_pages_by_doc_id(pages: List[Document]) -> Dict[str, List[Document]]:

    grouped: Dict[str, List[Document]] = {}
    for page in pages:
        doc_id: str = str(page.metadata.get("doc_id", ""))
        if not doc_id:
            continue
        grouped.setdefault(doc_id, []).append(page)
    return grouped


def _filter_duplicates_by_text_hash(pages: List[Document]) -> List[Document]:
    engine: Engine = _get_engine()
    collection_id: Optional[str] = _get_collection_id(engine)

    # Collection not created yet => nothing to deduplicate against.
    if collection_id is None:
        return pages

    grouped: Dict[str, List[Document]] = _group_pages_by_doc_id(pages)
    kept_pages: List[Document] = []

    for doc_id, doc_pages in grouped.items():
        original_filename: str = str(doc_pages[0].metadata.get("original_filename", "unknown"))
        new_text_hash: str = str(doc_pages[0].metadata.get("text_hash", ""))

        if not new_text_hash:
            kept_pages.extend(doc_pages)
            continue

        if _document_exists_by_text_hash(engine, collection_id=collection_id, text_hash=new_text_hash):
            print(f"Skipping duplicate document by text_hash: {original_filename} (doc_id={doc_id})")
            continue

        kept_pages.extend(doc_pages)

    return kept_pages


def clean_text_content(text: str) -> str:
    
    text = text.replace("\x00", "")
    # Replace single newlines with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def load_pdf_files(file_paths: List[str], original_filenames: Optional[List[str]] = None) -> List[Document]:
    if original_filenames is not None and len(original_filenames) != len(file_paths):
        raise ValueError("original_filenames must be the same length as file_paths")

    ingested_at: str = datetime.now(timezone.utc).isoformat()
    documents: List[Document] = []

    for index, file_path in enumerate(file_paths):
        original_filename: str = (
            original_filenames[index]
            if original_filenames is not None
            else os.path.basename(file_path)
        )
        try:
            loader: PyPDFLoader = PyPDFLoader(file_path)
            raw_docs: List[Document] = loader.load()
            for doc in raw_docs:
                doc.page_content = clean_text_content(doc.page_content)
                # Clean metadata from NUL characters
                doc.metadata = {
                    k: (v.replace("\x00", "") if isinstance(v, str) else v)
                    for k, v in doc.metadata.items()
                }

            text_hash: str = _compute_text_hash(raw_docs)
            doc_id: str = _make_doc_id(original_filename)
            _attach_document_metadata(
                raw_docs,
                doc_id=doc_id,
                original_filename=original_filename,
                text_hash=text_hash,
                version=1,
                ingested_at=ingested_at,
            )
            documents.extend(raw_docs)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    return documents


def load_pdf_documents(folder_path: str) -> List[Document]:

    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".pdf")
    ]
    return load_pdf_files(file_paths)


def split_documents(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    
    print("Initializing Semantic Chunker...")
    # Uses embedding similarity to determine where to split
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    return text_splitter.split_documents(documents)


def ingest_documents(
    source_files: Optional[List[str]] = None,
    source_filenames: Optional[List[str]] = None,
    cleanup: bool = True,
) -> Optional[PGVector]:

    print("Starting document ingestion process")

    documents: List[Document] = []

    if source_files:
        print(f"Ingesting {len(source_files)} specific files...")
        documents = load_pdf_files(source_files, original_filenames=source_filenames)
    else:
        if not os.path.exists(DOCS_FOLDER):
            print(f"Error: Folder '{DOCS_FOLDER}' not found.")
            print(f"Please create folder '{DOCS_FOLDER}' and place PDF files in it.")
            return None
        # Load documents 
        documents = load_pdf_documents(DOCS_FOLDER)

    if not documents:
        print("No valid documents found to ingest.")
        print("Ingestion process aborted.")
        return None

    print(f"Loaded {len(documents)} pages from PDF files.")

    if not cleanup:
        documents = _filter_duplicates_by_text_hash(documents)
        if not documents:
            print("All provided documents were skipped (duplicates).")
            return None
    
    # Initialize embeddings 
    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Split documents using Semantic Chunking
    splits: List[Document] = split_documents(documents, embeddings)
    print(f"Split text into {len(splits)} semantic chunks.")

    print(f"Saving vectors to PostgreSQL (Collection: {COLLECTION_NAME})...")

    try:
        # Save to DB
        vector_store: PGVector = PGVector.from_documents(
            embedding=embeddings,
            documents=splits,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            pre_delete_collection=cleanup,  
        )
        print("Document ingestion completed successfully.")
        return vector_store

    except Exception as e:
        print("Error connecting to database:")
        print(f"Details: {e}")
        return None


if __name__ == "__main__":
    ingest_documents()

