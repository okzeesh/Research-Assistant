#!/usr/bin/env python3
"""
Startup script for Research Assistant.
This script helps you start all required services and the application.
"""

import os
import sys
import subprocess
import time
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def start_elasticsearch():
    """Start Elasticsearch using Docker."""
    if not check_docker():
        logger.error("Docker is not installed or not running. Please install Docker first.")
        return False
    
    try:
        # Check if Elasticsearch container is already running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=research-assistant-elasticsearch'], 
                              capture_output=True, text=True)
        
        if 'research-assistant-elasticsearch' in result.stdout:
            logger.info("✅ Elasticsearch is already running")
            return True
        
        # Start Elasticsearch
        logger.info("🚀 Starting Elasticsearch...")
        subprocess.run([
            'docker', 'run', '-d',
            '--name', 'research-assistant-elasticsearch',
            '-p', '9200:9200',
            '-p', '9300:9300',
            '-e', 'discovery.type=single-node',
            '-e', 'xpack.security.enabled=false',
            '-e', 'ES_JAVA_OPTS=-Xms512m -Xmx512m',
            'docker.elastic.co/elasticsearch/elasticsearch:8.11.0'
        ], check=True)
        
        logger.info("✅ Elasticsearch started successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Elasticsearch: {e}")
        return False


def wait_for_elasticsearch():
    """Wait for Elasticsearch to be ready."""
    logger.info("⏳ Waiting for Elasticsearch to be ready...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get('http://localhost:9200', timeout=5)
            if response.status_code == 200:
                logger.info("✅ Elasticsearch is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        attempt += 1
        time.sleep(2)
        logger.info(f"⏳ Attempt {attempt}/{max_attempts}...")
    
    logger.error("❌ Elasticsearch failed to start within timeout")
    return False


def check_ollama():
    """Check if Ollama is running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama is running")
            return True
        else:
            logger.error("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.RequestException:
        logger.error("❌ Ollama is not running. Please start Ollama first.")
        logger.info("💡 To install Ollama, visit: https://ollama.ai")
        logger.info("💡 After installation, run: ollama pull llama2")
        return False


def install_dependencies():
    """Install Python dependencies."""
    logger.info("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def start_application():
    """Start the Research Assistant application."""
    logger.info("🚀 Starting Research Assistant...")
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Start the FastAPI application
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 'app.main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Research Assistant...")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")


def main():
    """Main startup function."""
    logger.info("🔬 Research Assistant Startup")
    logger.info("=" * 50)
    
    # Check if we're in the right directory
    if not Path('requirements.txt').exists():
        logger.error("❌ requirements.txt not found. Please run this script from the project root.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Start Elasticsearch
    if not start_elasticsearch():
        return 1
    
    # Wait for Elasticsearch
    if not wait_for_elasticsearch():
        return 1
    
    # Check Ollama
    if not check_ollama():
        logger.warning("⚠️  Ollama is not running. Some features may not work.")
        logger.info("💡 You can still start the application, but make sure to start Ollama later.")
    
    logger.info("🎉 All services are ready!")
    logger.info("📱 The application will be available at: http://localhost:8000")
    logger.info("📚 API documentation: http://localhost:8000/docs")
    logger.info("🌐 Web interface: http://localhost:8000/static/index.html")
    logger.info("=" * 50)
    
    # Start the application
    start_application()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 