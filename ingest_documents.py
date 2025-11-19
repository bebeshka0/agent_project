import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings  # Обновленный импорт
from langchain_core.documents import Document


# Configuration constants
DOCS_FOLDER: str = "documents"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CONNECTION_STRING: str = "postgresql://postgres:mysecretpassword@localhost:5432/vector_db"  
COLLECTION_NAME: str = "ml_articles_collection"


def load_pdf_documents(folder_path: str) -> List[Document]:

    loaders: List[PyPDFLoader] = [
        PyPDFLoader(os.path.join(folder_path, filename))
        for filename in os.listdir(folder_path)
        if filename.endswith(".pdf")
    ]

    documents: List[Document] = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            file_path: str = getattr(loader, "file_path", "unknown")
            print(f"Error loading file {file_path}: {e}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def clean_documents(documents: List[Document]) -> List[Document]:

    cleaned_documents: List[Document] = []

    for document in documents:
        cleaned_content: str = document.page_content.replace("\x00", "")

        cleaned_metadata: dict = {
            key: (value.replace("\x00", "") if isinstance(value, str) else value)
            for key, value in document.metadata.items()
        }

        cleaned_documents.append(
            Document(page_content=cleaned_content, metadata=cleaned_metadata)
        )

    return cleaned_documents


def ingest_documents() -> Optional[PGVector]:

    print("Starting document ingestion process")

    # Check if docs folder exists
    if not os.path.exists(DOCS_FOLDER):
        print(f"Error: Folder '{DOCS_FOLDER}' not found.")
        print(f"Please create folder '{DOCS_FOLDER}' and place PDF files in it.")
        return None

    # Load PDF documents
    documents: List[Document] = load_pdf_documents(DOCS_FOLDER)

    if not documents:
        print(f"No PDF documents found in folder '{DOCS_FOLDER}'.")
        print("Ingestion process aborted.")
        return None

    print(f"Loaded {len(documents)} pages from PDF files.")

    # Split documents into chunks
    splits: List[Document] = split_documents(documents)
    print(f"Split text into {len(splits)} chunks.")

    # Clean documents from NUL characters before writing to PostgreSQL
    cleaned_splits: List[Document] = clean_documents(splits)

    # Initialize embedding model
    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Store vectors in PostgreSQL
    print(f"Saving vectors to PostgreSQL (Collection: {COLLECTION_NAME})...")

    try:
        # This method creates table if it does not exist and adds documents
        vector_store: PGVector = PGVector.from_documents(
            embedding=embeddings,
            documents=cleaned_splits,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            # pre_delete_collection=True  # Uncomment to delete and recreate collection
        )
        print("Document ingestion completed successfully.")
        return vector_store

    except Exception as e:
        print("Error connecting to database:")
        print(f"Details: {e}")
        return None


if __name__ == "__main__":
    ingest_documents()

