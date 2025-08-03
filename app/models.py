from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class UploadResponse(BaseModel):
    """Response model for file upload."""
    message: str
    file_id: str
    filename: str
    file_size: int
    processing_status: str


class SummaryRequest(BaseModel):
    """Request model for paper summarization."""
    file_id: str
    summary_type: str = "general"  # general, detailed, key_points


class SummaryResponse(BaseModel):
    """Response model for paper summary."""
    file_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    summary: str
    key_points: List[str]
    summary_type: str
    generated_at: datetime


class QueryRequest(BaseModel):
    """Request model for user queries."""
    query: str
    file_id: Optional[str] = None
    context_type: str = "paper"  # paper, literature, general


class QueryResponse(BaseModel):
    """Response model for query responses."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    generated_at: datetime


class RelatedPapersRequest(BaseModel):
    """Request model for finding related papers."""
    query: str
    limit: int = 10


class RelatedPapersResponse(BaseModel):
    """Response model for related papers."""
    query: str
    papers: List[Dict[str, Any]]
    total_found: int


class PaperMetadata(BaseModel):
    """Model for paper metadata."""
    title: str
    authors: List[str]
    abstract: str
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = []
    file_id: str
    uploaded_at: datetime 