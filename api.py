import os
import shutil
import tempfile
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from rag_agent import build_rag_chain
from router_agent import build_router_chain
from tutor_agent import build_tutor_chain
from ingest_documents import ingest_documents

# Global state for chains
chains = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing agents...")
    try:
        chains["tutor"] = build_tutor_chain()
        chains["rag"] = build_rag_chain()
        chains["router"] = build_router_chain()
        print("Agents initialized successfully.")
    except Exception as e:
        print(f"Error initializing agents: {e}")
    
    yield
    
    # Clean up resources
    chains.clear()

app = FastAPI(lifespan=lifespan, title="ML Tutor API")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    source: str
    context: Optional[str] = None

class IngestResponse(BaseModel):
    message: str
    status: str

def process_files_background(file_paths: List[str], original_filenames: List[str]) -> None:

    try:
        print(f"Background task started: Processing {len(file_paths)} files...")
        # Ingest documents without clearing existing collection
        ingest_documents(
            source_files=file_paths,
            source_filenames=original_filenames,
            cleanup=False,
        )
        print("Background task completed: Documents ingested.")
        
        chains["rag"] = build_rag_chain()
        
    except Exception as e:
        print(f"Error in background ingestion task: {e}")
    finally:
        # Cleanup temporary files
        for path in file_paths:
            try:
                os.remove(path)
                print(f"Deleted temp file: {path}")
            except OSError as e:
                print(f"Error deleting temp file {path}: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    question = request.question
    
    if not chains:
        raise HTTPException(status_code=503, detail="Agents not initialized")

    # Routing
    try:
        route = chains["router"].invoke(question)
    except Exception as e:
        print(f"Router error: {e}. Falling back to TUTOR.")
        route = "TUTOR"

    if route == "RAG":
        try:
            result = chains["rag"].invoke(question)
            return ChatResponse(
                answer=result.get("answer", ""),
                source="RAG",
                context=result.get("context", "")
            )
        except Exception as e:
            print(f"RAG Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing RAG request: {str(e)}")
    else:
        try:
            answer = chains["tutor"].invoke(question)
            return ChatResponse(
                answer=answer,
                source="TUTOR",
                context=None
            )
        except Exception as e:
             print(f"Tutor Error: {e}")
             raise HTTPException(status_code=500, detail=f"Error processing Tutor request: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_file_paths: List[str] = []
    original_filenames: List[str] = []
    
    try:
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue
                
            # Create a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_file_paths.append(tmp_file.name)
                original_filenames.append(file.filename)
        
        if not temp_file_paths:
             raise HTTPException(status_code=400, detail="No valid PDF files found")

        background_tasks.add_task(process_files_background, temp_file_paths, original_filenames)

        return IngestResponse(
            message=f"Started processing {len(temp_file_paths)} documents in the background.",
            status="processing"
        )

    except Exception as e:
        for path in temp_file_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "agents_loaded": bool(chains)}
