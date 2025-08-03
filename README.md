# Research Assistant

An AI-powered research assistant that summarizes academic papers, answers questions, and retrieves relevant literature using local LLMs and semantic search.

## 🚀 Features

- **PDF Paper Upload & Processing**: Upload academic papers in PDF format
- **Intelligent Text Extraction**: Extract and chunk text using PyMuPDF and pdfminer
- **Semantic Search**: Find relevant papers using Elasticsearch and sentence transformers
- **Multi-Agent System**: CrewAI-powered agents for retrieval, summarization, and Q&A
- **Local LLM Integration**: Uses Ollama for local language model inference
- **RESTful API**: FastAPI-based API with comprehensive endpoints

## 🏗️ Architecture

```
User ──► FastAPI ──► CrewAI Agents ──► Ollama (LLMs)
                     │
                     └────► Elasticsearch (Semantic Retrieval)
```

### System Modules

1. **Paper Ingestion Module**: PDF processing and text extraction
2. **Embedding & Indexing Module**: Text vectorization and Elasticsearch storage
3. **Multi-Agent System (CrewAI)**:
   - **Retriever Agent**: Fetches relevant papers and content
   - **Summarizer Agent**: Generates concise summaries
   - **Answer Agent**: Provides contextual answers

## 📋 Prerequisites

- Python 3.8+
- Docker (for Elasticsearch)
- Ollama (for local LLM)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Research-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Elasticsearch

Using Docker:

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### 4. Set Up Ollama

Install Ollama from [ollama.ai](https://ollama.ai) and pull a model:

```bash
# Install Ollama (follow instructions on website)
# Then pull a model
ollama pull llama2
```

### 5. Environment Configuration

Create a `.env` file (optional):

```env
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=research_papers
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
UPLOAD_DIR=uploads
MAX_FILE_SIZE=52428800
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🚀 Usage

### 1. Start the Application

```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

### 2. API Endpoints

#### Upload Paper

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_paper.pdf"
```

#### Generate Summary

```bash
curl -X POST "http://localhost:8000/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your_file_id",
    "summary_type": "general"
  }'
```

#### Ask Questions

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings of this paper?",
    "file_id": "your_file_id"
  }'
```

#### Find Related Papers

```bash
curl -X POST "http://localhost:8000/related" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications",
    "limit": 10
  }'
```

### 3. API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📁 Project Structure

```
Research-Assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic models
│   └── modules/
│       ├── __init__.py
│       ├── paper_ingestion.py    # PDF processing
│       ├── embedding_indexing.py # Vector search
│       └── crew_agents.py        # Multi-agent system
├── uploads/                 # Uploaded files
├── requirements.txt         # Dependencies
└── README.md
```

## 🔧 Configuration

### Elasticsearch Settings

- **URL**: `http://localhost:9200`
- **Index**: `research_papers`
- **Vector Dimension**: 384 (all-MiniLM-L6-v2)

### Ollama Settings

- **Base URL**: `http://localhost:11434`
- **Model**: `llama2` (configurable)

### Text Processing

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max File Size**: 50MB

## 🚀 Deployment

### Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t research-assistant .

# Run container
docker run -p 8000:8000 research-assistant
```

### Cloud Deployment

#### Railway

1. Connect your GitHub repository
2. Set environment variables
3. Deploy automatically

#### Render.com

1. Create a new Web Service
2. Connect your repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### AWS EC2

1. Launch EC2 instance
2. Install Docker and dependencies
3. Run with docker-compose or directly

## 🔍 Troubleshooting

### Common Issues

1. **Elasticsearch Connection Error**

   - Ensure Elasticsearch is running on port 9200
   - Check if security is disabled for development

2. **Ollama Connection Error**

   - Verify Ollama is running on port 11434
   - Ensure the specified model is downloaded

3. **PDF Processing Errors**

   - Check if PyMuPDF is properly installed
   - Verify PDF file is not corrupted

4. **Memory Issues**
   - Reduce chunk size for large documents
   - Use smaller embedding models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for multi-agent orchestration
- [Ollama](https://ollama.ai) for local LLM inference
- [Elasticsearch](https://elastic.co) for semantic search
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
