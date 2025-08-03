import json
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingIndexingModule:
    """Module for generating embeddings and indexing in Elasticsearch."""
    
    def __init__(self):
        self.es_client = Elasticsearch([settings.ELASTICSEARCH_URL])
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_name = settings.ELASTICSEARCH_INDEX
        
        # Initialize index
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Create Elasticsearch index with proper mapping if it doesn't exist."""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "file_id": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "authors": {"type": "keyword"},
                            "abstract": {"type": "text"},
                            "content": {"type": "text"},
                            "content_vector": {
                                "type": "dense_vector",
                                "dims": 384  # all-MiniLM-L6-v2 dimension
                            },
                            "metadata": {"type": "object"},
                            "uploaded_at": {"type": "date"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                
                self.es_client.indices.create(
                    index=self.index_name,
                    body=mapping
                )
                logger.info(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Index a processed paper in Elasticsearch."""
        try:
            file_id = paper_data['file_id']
            chunks = paper_data['chunks']
            metadata = paper_data['metadata']
            
            # Generate embeddings for all chunks
            embeddings = self.generate_embeddings(chunks)
            
            # Prepare documents for indexing
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "file_id": file_id,
                    "chunk_id": f"{file_id}_chunk_{i}",
                    "title": metadata.get('title', ''),
                    "authors": metadata.get('authors', []),
                    "abstract": metadata.get('abstract', ''),
                    "content": chunk,
                    "content_vector": embedding,
                    "metadata": metadata,
                    "uploaded_at": paper_data.get('uploaded_at')
                }
                documents.append(doc)
            
            # Bulk index documents
            bulk_data = []
            for doc in documents:
                bulk_data.append({"index": {"_index": self.index_name}})
                bulk_data.append(doc)
            
            if bulk_data:
                response = self.es_client.bulk(body=bulk_data, refresh=True)
                if response.get('errors'):
                    logger.error(f"Bulk indexing errors: {response}")
                    return False
                
                logger.info(f"Successfully indexed {len(documents)} chunks for file {file_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error indexing paper: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 10, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content using semantic similarity."""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Build search query
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": top_k
            }
            
            # Add file filter if specified
            if file_id:
                search_body["query"]["script_score"]["query"] = {
                    "term": {"file_id": file_id}
                }
            
            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    'file_id': hit['_source']['file_id'],
                    'chunk_id': hit['_source']['chunk_id'],
                    'content': hit['_source']['content'],
                    'title': hit['_source']['title'],
                    'authors': hit['_source']['authors'],
                    'abstract': hit['_source']['abstract']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            return []
    
    def get_paper_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific paper."""
        try:
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"file_id": file_id}},
                    "size": 1000,
                    "sort": [{"chunk_id": {"order": "asc"}}]
                }
            )
            
            chunks = []
            for hit in response['hits']['hits']:
                chunks.append({
                    'chunk_id': hit['_source']['chunk_id'],
                    'content': hit['_source']['content'],
                    'title': hit['_source']['title'],
                    'authors': hit['_source']['authors']
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving paper chunks: {e}")
            return []
    
    def delete_paper(self, file_id: str) -> bool:
        """Delete all chunks for a specific paper."""
        try:
            response = self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"file_id": file_id}}}
            )
            
            deleted_count = response['deleted']
            logger.info(f"Deleted {deleted_count} chunks for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting paper: {e}")
            return False 