import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Elasticsearch Configuration
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "research_papers"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Text Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API Configuration
    API_TITLE: str = "Research Assistant API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered research assistant for academic papers"
    
    class Config:
        env_file = ".env"


settings = Settings() 