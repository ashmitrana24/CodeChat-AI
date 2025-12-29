"""
RAG Codebase Analysis System - REST API

FastAPI application providing endpoints for:
- Loading a repository
- Asking questions about code
- Checking system status
"""
import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os

# Import modules
from modules.ingestion import CodebaseIngester
from modules.chunking import CodeChunker
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore
from modules.question_processor import QuestionProcessor
from modules.rag_generator import RAGGenerator, RAGResponse


# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="RAG Codebase Analysis API",
    description="Analyze codebases and answer questions using RAG",
    version="1.0.0"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")


# Root redirect to frontend
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the frontend."""
    return RedirectResponse(url="/static/index.html")


# =============================================================================
# Request/Response Models
# =============================================================================

class LoadRepositoryRequest(BaseModel):
    """Request to load a repository."""
    path: str = Field(..., description="Absolute or relative path to the repository")


class LoadRepositoryResponse(BaseModel):
    """Response after loading a repository."""
    success: bool
    message: str
    stats: dict


class AskQuestionRequest(BaseModel):
    """Request to ask a question."""
    question: str = Field(..., description="Natural language question about the code")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")


class AskQuestionResponse(BaseModel):
    """Response with answer and sources."""
    answer: str
    source_files: List[str]
    chunks_used: int


class StatusResponse(BaseModel):
    """System status response."""
    repository_loaded: bool
    repository_path: Optional[str]
    stats: Optional[dict]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Holds the application state."""
    
    def __init__(self):
        self.ingester = CodebaseIngester()
        self.chunker = CodeChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.question_processor: Optional[QuestionProcessor] = None
        self.rag_generator: Optional[RAGGenerator] = None
        
        self.repository_loaded = False
        self.repository_path: Optional[str] = None
        self.load_stats: Optional[dict] = None


# Global application state
state = AppState()


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is running."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get the current system status."""
    stats = None
    if state.repository_loaded:
        stats = state.vector_store.get_stats()
    
    return StatusResponse(
        repository_loaded=state.repository_loaded,
        repository_path=state.repository_path,
        stats=stats
    )


@app.post("/load", response_model=LoadRepositoryResponse, tags=["Repository"])
async def load_repository(request: LoadRepositoryRequest):
    """
    Load a repository for analysis.
    
    This will:
    1. Scan the directory for supported source files
    2. Split files into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    try:
        # Resolve the path
        repo_path = os.path.abspath(request.path)
        
        if not os.path.exists(repo_path):
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist: {repo_path}"
            )
        
        if not os.path.isdir(repo_path):
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {repo_path}"
            )
        
        # Clear previous state
        state.vector_store.clear()
        
        # Step 1: Ingest files
        print(f"[*] Ingesting files from: {repo_path}")
        source_files = state.ingester.ingest(repo_path)
        
        if not source_files:
            raise HTTPException(
                status_code=400,
                detail="No supported source files found in the repository"
            )
        
        ingest_stats = state.ingester.get_stats(source_files)
        print(f"   Found {ingest_stats['total_files']} files")
        
        # Step 2: Chunk files
        print("[*] Chunking files...")
        chunks = state.chunker.chunk_files(source_files)
        chunk_stats = state.chunker.get_stats(chunks)
        print(f"   Created {chunk_stats['total_chunks']} chunks")
        
        # Step 3: Generate embeddings
        print("[*] Generating embeddings...")
        embeddings = state.embedding_generator.embed_chunks(chunks)
        print(f"   Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database
        print("[*] Storing in vector database...")
        state.vector_store.add_embeddings(embeddings, chunks)
        
        # Initialize processors
        state.question_processor = QuestionProcessor(
            state.embedding_generator,
            state.vector_store
        )
        state.rag_generator = RAGGenerator(state.question_processor)
        
        # Update state
        state.repository_loaded = True
        state.repository_path = repo_path
        state.load_stats = {
            "files": ingest_stats,
            "chunks": chunk_stats,
            "vectors": state.vector_store.get_stats()
        }
        
        print("[+] Repository loaded successfully!")
        
        return LoadRepositoryResponse(
            success=True,
            message=f"Successfully loaded {ingest_stats['total_files']} files with {chunk_stats['total_chunks']} chunks",
            stats=state.load_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading repository: {str(e)}"
        )


@app.post("/ask", response_model=AskQuestionResponse, tags=["Q&A"])
async def ask_question(request: AskQuestionRequest):
    """
    Ask a question about the loaded codebase.
    
    The question will be:
    1. Converted to an embedding
    2. Used to find similar code chunks
    3. Passed with context to the LLM
    4. Returned with file references
    """
    if not state.repository_loaded:
        raise HTTPException(
            status_code=400,
            detail="No repository loaded. Call /load first."
        )
    
    if not state.rag_generator:
        raise HTTPException(
            status_code=500,
            detail="RAG generator not initialized"
        )
    
    try:
        # Update top_k if provided
        if request.top_k:
            state.question_processor.top_k = request.top_k
        
        # Generate response
        response: RAGResponse = state.rag_generator.generate(request.question)
        
        return AskQuestionResponse(
            answer=response.answer,
            source_files=response.source_files,
            chunks_used=response.chunks_used
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


@app.post("/search", tags=["Q&A"])
async def search_code(request: AskQuestionRequest):
    """
    Search for relevant code chunks without generating an answer.
    
    Useful for exploring the codebase or debugging retrieval.
    """
    if not state.repository_loaded:
        raise HTTPException(
            status_code=400,
            detail="No repository loaded. Call /load first."
        )
    
    try:
        if request.top_k:
            state.question_processor.top_k = request.top_k
        
        results = state.question_processor.process(request.question)
        
        return {
            "results": [
                {
                    "file": r.chunk.relative_path,
                    "score": r.score,
                    "rank": r.rank,
                    "language": r.chunk.language,
                    "content": r.chunk.content[:500] + "..." if len(r.chunk.content) > 500 else r.chunk.content
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching: {str(e)}"
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
