"""
Backend System Validator and Startup Script
Tests and validates the backend system before full deployment
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ required")
        return False
    return True

def check_directories():
    """Check and create required directories"""
    required_dirs = [
        "uploads",
        "static", 
        "temp",
        "reports"
    ]
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory exists: {directory}")
    
    return True

def check_imports():
    """Check if essential imports work"""
    try:
        import fastapi
        import uvicorn
        import aiofiles
        logger.info("Essential imports successful")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Run: pip install -r requirements_minimal.txt")
        return False

def test_working_api():
    """Test if the working API can be imported"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        import working_api
        logger.info("Working API imported successfully")
        return True
    except Exception as e:
        logger.error(f"Working API import failed: {e}")
        return False

def test_original_modules():
    """Test original modules with error tolerance"""
    results = {}
    
    modules_to_test = [
        ("utils.database", "Database"),
        ("utils.file_manager", "FileManager"),
        ("utils.report_generator", "ReportGenerator"),
        ("models.model_manager", "ModelManager")
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            logger.info(f"✓ {module_name}.{class_name} - OK")
            results[module_name] = True
        except Exception as e:
            logger.warning(f"✗ {module_name}.{class_name} - Failed: {e}")
            results[module_name] = False
    
    return results

async def test_api_endpoints():
    """Test API endpoints programmatically"""
    try:
        from fastapi.testclient import TestClient
        import working_api
        
        client = TestClient(working_api.app)
        
        # Test root endpoint
        response = client.get("/")
        if response.status_code == 200:
            logger.info("✓ Root endpoint working")
        else:
            logger.error(f"✗ Root endpoint failed: {response.status_code}")
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            logger.info("✓ Health endpoint working")
        else:
            logger.error(f"✗ Health endpoint failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"API endpoint test failed: {e}")
        return False

def main():
    """Main validation and startup routine"""
    logger.info("🔍 Starting Backend System Validation...")
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_directories():
        sys.exit(1)
    
    if not check_imports():
        logger.error("❌ Essential imports failed")
        logger.info("💡 Install minimal requirements: pip install -r requirements_minimal.txt")
        sys.exit(1)
    
    # Test working API
    if not test_working_api():
        logger.error("❌ Working API failed to load")
        sys.exit(1)
    
    # Test original modules (with error tolerance)
    logger.info("🧪 Testing original modules...")
    module_results = test_original_modules()
    
    working_modules = sum(module_results.values())
    total_modules = len(module_results)
    
    logger.info(f"📊 Module Status: {working_modules}/{total_modules} working")
    
    # Test API endpoints
    logger.info("🌐 Testing API endpoints...")
    asyncio.run(test_api_endpoints())
    
    logger.info("✅ Backend validation complete!")
    logger.info("🚀 Ready to start server with: uvicorn working_api:app --reload --host 0.0.0.0 --port 8000")
    
    return True

if __name__ == "__main__":
    main()
