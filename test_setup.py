#!/usr/bin/env python3
"""
Test script to verify Research Assistant setup and dependencies.
"""

import sys
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_elasticsearch():
    """Test Elasticsearch connection."""
    try:
        response = requests.get("http://localhost:9200", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Elasticsearch is running")
            return True
        else:
            logger.error(f"❌ Elasticsearch returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to Elasticsearch: {e}")
        return False


def test_ollama():
    """Test Ollama connection."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama is running")
            return True
        else:
            logger.error(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}")
        return False


def test_api():
    """Test Research Assistant API."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Research Assistant API is running")
            return True
        else:
            logger.error(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return False


def test_dependencies():
    """Test Python dependencies."""
    try:
        import fastapi
        import uvicorn
        import fitz  # PyMuPDF
        import sentence_transformers
        import elasticsearch
        import crewai
        import langchain
        import ollama
        logger.info("✅ All Python dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🔍 Testing Research Assistant Setup...")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("-" * 50)
    
    tests = [
        ("Python Dependencies", test_dependencies),
        ("Elasticsearch", test_elasticsearch),
        ("Ollama", test_ollama),
        ("Research Assistant API", test_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Error testing {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your Research Assistant is ready to use.")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Please check the setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 