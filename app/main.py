import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles

from app.config import settings
from app.models import (
    UploadResponse, SummaryRequest, SummaryResponse, 
    QueryRequest, QueryResponse, RelatedPapersRequest, RelatedPapersResponse
)
from app.modules.paper_ingestion import PaperIngestionModule
from app.modules.embedding_indexing import EmbeddingIndexingModule
from app.modules.crew_agents import ResearchAssistantCrew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
paper_ingestion = PaperIngestionModule()
embedding_indexing = EmbeddingIndexingModule()
crew_system = ResearchAssistantCrew()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# In-memory storage for file metadata (in production, use a database)
file_metadata = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Research Assistant API...")
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@app.post("/upload", response_model=UploadResponse)
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a PDF paper."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size too large")
        
        # Read file content
        file_content = await file.read()
        
        # Save file
        file_path = paper_ingestion.save_uploaded_file(file_content, file.filename)
        
        # Process paper in background
        background_tasks.add_task(process_paper_background, file_path, file.filename, file.size)
        
        # Generate file ID from path
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        
        return UploadResponse(
            message="File uploaded successfully. Processing in background.",
            file_id=file_id,
            filename=file.filename,
            file_size=file.size,
            processing_status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_paper_background(file_path: str, filename: str, file_size: int):
    """Background task to process uploaded paper."""
    try:
        # Process paper
        paper_data = paper_ingestion.process_paper(file_path)
        paper_data['uploaded_at'] = datetime.now()
        paper_data['filename'] = filename
        paper_data['file_size'] = file_size
        
        # Index in Elasticsearch
        success = embedding_indexing.index_paper(paper_data)
        
        if success:
            # Store metadata
            file_id = paper_data['file_id']
            file_metadata[file_id] = {
                'filename': filename,
                'file_size': file_size,
                'uploaded_at': paper_data['uploaded_at'],
                'metadata': paper_data['metadata'],
                'processing_status': 'completed'
            }
            logger.info(f"Successfully processed paper: {file_id}")
        else:
            logger.error(f"Failed to index paper: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing paper in background: {e}")


@app.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """Generate a summary of an uploaded paper."""
    try:
        file_id = request.file_id
        
        # Check if file exists and is processed
        if file_id not in file_metadata:
            raise HTTPException(status_code=404, detail="File not found or not yet processed")
        
        if file_metadata[file_id]['processing_status'] != 'completed':
            raise HTTPException(status_code=400, detail="File is still being processed")
        
        # Generate summary using CrewAI
        result = crew_system.generate_summary(file_id, request.summary_type)
        
        metadata = file_metadata[file_id]['metadata']
        
        return SummaryResponse(
            file_id=file_id,
            title=metadata.get('title', ''),
            authors=metadata.get('authors', []),
            abstract=metadata.get('abstract', ''),
            summary=result['summary'],
            key_points=[],  # Could be extracted separately
            summary_type=request.summary_type,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """Answer a user query about papers."""
    try:
        # Answer query using CrewAI
        result = crew_system.answer_query(request.query, request.file_id)
        
        return QueryResponse(
            query=request.query,
            answer=result['answer'],
            sources=[],  # Could be enhanced to include source papers
            confidence_score=0.8,  # Placeholder
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/related", response_model=RelatedPapersResponse)
async def find_related_papers(request: RelatedPapersRequest):
    """Find papers related to a query."""
    try:
        # Search for related papers
        results = embedding_indexing.search_similar(request.query, top_k=request.limit)
        
        papers = []
        for result in results:
            paper = {
                'file_id': result['file_id'],
                'title': result['title'],
                'authors': result['authors'],
                'abstract': result['abstract'],
                'relevance_score': result['score'],
                'content_preview': result['content'][:200] + "..."
            }
            papers.append(paper)
        
        return RelatedPapersResponse(
            query=request.query,
            papers=papers,
            total_found=len(papers)
        )
        
    except Exception as e:
        logger.error(f"Error finding related papers: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/files")
async def list_files():
    """List all uploaded files."""
    return {
        "files": [
            {
                "file_id": file_id,
                "filename": metadata['filename'],
                "file_size": metadata['file_size'],
                "uploaded_at": metadata['uploaded_at'],
                "processing_status": metadata['processing_status'],
                "metadata": metadata['metadata']
            }
            for file_id, metadata in file_metadata.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 