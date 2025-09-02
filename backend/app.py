"""
FastAPI Backend for Digital Deepfake Detection System
Web application for detecting AI-generated content in media files
"""

import os
import logging
from datetime import datetime
from typing import List
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Custom imports
from api.video_detector import VideoDetector
from api.audio_detector import AudioDetector
from api.image_detector import ImageDetector
from api.live_detector import LiveDetector
from utils.file_manager import FileManager
from utils.database import Database
from utils.report_generator import ReportGenerator
from models.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Digital Deepfake Detection API",
    description="API for detecting AI-generated content in videos, images, and audio files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
file_manager = FileManager()
database = Database()
model_manager = ModelManager()
report_generator = ReportGenerator()

# Detection components
video_detector = VideoDetector(model_manager)
audio_detector = AudioDetector(model_manager)
image_detector = ImageDetector(model_manager)
live_detector = LiveDetector(model_manager)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup"""
    logger.info("Starting Digital Deepfake Detection System...")
    
    # Initialize database
    await database.init_db()
    
    # Load AI models
    await model_manager.load_models()
    
    # Create upload directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("System initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Digital Deepfake Detection API",
        "status": "active",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": model_manager.models_loaded,
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and analyze video for deepfake detection"""
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded file
        file_path = await file_manager.save_upload(file)
        task_id = str(uuid.uuid4())
        
        # Process video asynchronously
        result = await video_detector.analyze_video(file_path, task_id)
        
        # Save results to database
        await database.save_analysis(task_id, "video", result)
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error("Video upload error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and analyze audio for deepfake detection"""
    try:
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        file_path = await file_manager.save_upload(file)
        task_id = str(uuid.uuid4())
        
        result = await audio_detector.analyze_audio(file_path, task_id)
        await database.save_analysis(task_id, "audio", result)
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error("Audio upload error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and analyze image for deepfake detection"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        file_path = await file_manager.save_upload(file)
        task_id = str(uuid.uuid4())
        
        result = await image_detector.analyze_image(file_path, task_id)
        await database.save_analysis(task_id, "image", result)
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        logger.error("Image upload error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/batch/process")
async def batch_process(files: List[UploadFile] = File(...)):
    """Process multiple files in batch"""
    try:
        batch_id = str(uuid.uuid4())
        results = []
        
        for file in files:
            file_path = await file_manager.save_upload(file)
            task_id = str(uuid.uuid4())
            
            # Determine file type and process accordingly
            if file.content_type.startswith('video/'):
                result = await video_detector.analyze_video(file_path, task_id)
            elif file.content_type.startswith('audio/'):
                result = await audio_detector.analyze_audio(file_path, task_id)
            elif file.content_type.startswith('image/'):
                result = await image_detector.analyze_image(file_path, task_id)
            else:
                continue
            
            await database.save_analysis(task_id, "batch", result)
            results.append({
                "task_id": task_id,
                "filename": file.filename,
                "result": result
            })
        
        return {
            "batch_id": batch_id,
            "total_files": len(files),
            "processed_files": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error("Batch processing error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Retrieve analysis results by task ID"""
    try:
        result = await database.get_analysis(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        return result
    except Exception as e:
        logger.error("Get results error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/report/generate/{task_id}")
async def generate_report(task_id: str, report_format: str = "pdf"):
    """Generate downloadable report for analysis results"""
    try:
        result = await database.get_analysis(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if report_format.lower() == "pdf":
            report_path = await report_generator.generate_pdf_report(result)
        elif report_format.lower() == "csv":
            report_path = await report_generator.generate_csv_report(result)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        return FileResponse(
            path=report_path,
            filename=f"deepfake_analysis_{task_id}.{report_format}",
            media_type=f"application/{report_format}"
        )
        
    except Exception as e:
        logger.error("Report generation error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.websocket("/ws/live/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for live detection"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            # Process live detection (placeholder)
            result = await live_detector.process_live_data(data)
            
            # Send results back to client
            await manager.send_personal_message(result, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client %s disconnected", client_id)

@app.get("/api/demo/samples")
async def get_demo_samples():
    """Get list of demo samples for testing"""
    try:
        samples_dir = "demo_data"
        if not os.path.exists(samples_dir):
            return {"samples": []}
        
        samples = []
        for filename in os.listdir(samples_dir):
            file_path = os.path.join(samples_dir, filename)
            if os.path.isfile(file_path):
                samples.append({
                    "filename": filename,
                    "path": f"/api/demo/download/{filename}",
                    "type": filename.split('.')[-1].lower()
                })
        
        return {"samples": samples}
        
    except Exception as e:
        logger.error("Demo samples error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/demo/download/{filename}")
async def download_demo_file(filename: str):
    """Download demo file"""
    file_path = os.path.join("demo_data", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path=file_path, filename=filename)

@app.get("/api/stats/dashboard")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        stats = await database.get_analysis_stats()
        return stats
    except Exception as e:
        logger.error("Dashboard stats error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
