import os
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.layout import LAParams
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class PaperIngestionModule:
    """Module for processing and ingesting academic papers."""
    
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            # Fallback to pdfminer
            return self._extract_text_with_pdfminer(file_path)
    
    def _extract_text_with_pdfminer(self, file_path: str) -> str:
        """Fallback text extraction using pdfminer."""
        try:
            laparams = LAParams()
            text = pdfminer_extract(file_path, laparams=laparams)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text with pdfminer: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if overlap is None:
            overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract basic metadata from paper text."""
        metadata = {
            'title': '',
            'authors': [],
            'abstract': '',
            'keywords': []
        }
        
        lines = text.split('\n')
        
        # Try to extract title (usually first few lines)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                if not any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'doi:', 'http']):
                    metadata['title'] = line
                    break
        
        # Try to extract abstract
        abstract_start = -1
        for i, line in enumerate(lines):
            if 'abstract' in line.lower():
                abstract_start = i
                break
        
        if abstract_start != -1:
            abstract_lines = []
            for line in lines[abstract_start + 1:]:
                line = line.strip()
                if line and not line.startswith('Keywords:') and not line.startswith('1.'):
                    abstract_lines.append(line)
                elif line.startswith('Keywords:') or line.startswith('1.'):
                    break
            metadata['abstract'] = ' '.join(abstract_lines)
        
        return metadata
    
    def process_paper(self, file_path: str) -> Dict[str, Any]:
        """Process a paper and return structured data."""
        file_id = str(uuid.uuid4())
        
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        
        # Extract metadata
        metadata = self.extract_metadata(text)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        return {
            'file_id': file_id,
            'original_text': text,
            'chunks': chunks,
            'metadata': metadata,
            'file_path': file_path,
            'file_size': os.path.getsize(file_path)
        }
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return file path."""
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(filename)[1]
        saved_filename = f"{file_id}{file_extension}"
        file_path = os.path.join(self.upload_dir, saved_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return file_path 