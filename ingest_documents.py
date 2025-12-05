import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

load_dotenv()

DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "documents")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CONNECTION_STRING: str = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql://postgres:mysecretpassword@localhost:5432/vector_db",
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ml_articles_collection")


def clean_text_content(text: str) -> str:
    
    text = text.replace("\x00", "")
    # Replace single newlines with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def load_pdf_files(file_paths: List[str]) -> List[Document]:

    loaders: List[PyPDFLoader] = [PyPDFLoader(fp) for fp in file_paths]

    documents: List[Document] = []
    for loader in loaders:
        try:
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.page_content = clean_text_content(doc.page_content)
                # Clean metadata from NUL characters
                doc.metadata = {
                    k: (v.replace("\x00", "") if isinstance(v, str) else v)
                    for k, v in doc.metadata.items()
                }
            documents.extend(raw_docs)
        except Exception as e:
            file_path: str = getattr(loader, "file_path", "unknown")
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
    source_files: Optional[List[str]] = None, cleanup: bool = True
) -> Optional[PGVector]:

    print("Starting document ingestion process")

    documents: List[Document] = []

    if source_files:
        print(f"Ingesting {len(source_files)} specific files...")
        documents = load_pdf_files(source_files)
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

