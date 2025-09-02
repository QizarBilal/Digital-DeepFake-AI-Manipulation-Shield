"""
Enhanced API endpoints for multi-modal deepfake detection
Integrates with existing FastAPI backend and provides comprehensive detection services
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import asyncio
import json
from datetime import datetime
import uuid

# Import our detection models
from models.multimodal_fusion import MultiModalDeepfakeDetector
from models.advanced_image_detector import ImageDeepfakeDetector
from models.advanced_video_detector import VideoDeepfakeDetector 
from models.advanced_audio_detector import AudioDeepfakeDetector
from sample_datasets.dataset_manager import SampleDatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Deepfake Detection API",
    description="Multi-modal deepfake detection system with image, video, and audio analysis",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection models globally
multimodal_detector = None
image_detector = None
video_detector = None
audio_detector = None
dataset_manager = None

# Upload directory configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Active analyses storage (in production, use database)
active_analyses = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global multimodal_detector, image_detector, video_detector, audio_detector, dataset_manager
    
    try:
        logger.info("Initializing detection models...")
        
        # Initialize individual detectors
        image_detector = ImageDeepfakeDetector()
        video_detector = VideoDeepfakeDetector()
        audio_detector = AudioDeepfakeDetector()
        
        # Initialize multi-modal detector
        multimodal_detector = MultiModalDeepfakeDetector()
        
        # Initialize dataset manager
        dataset_manager = SampleDatasetManager()
        
        logger.info("All models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Continue with limited functionality

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Deepfake Detection API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Multi-modal detection (image + video + audio)",
            "Individual modality analysis",
            "Real-time confidence scoring",
            "Anomaly visualization",
            "Sample dataset testing"
        ],
        "endpoints": {
            "single_modality": "/detect/{modality}",
            "multi_modal": "/detect/multimodal",
            "batch_analysis": "/detect/batch",
            "sample_demo": "/demo/samples",
            "live_analysis": "/analyze/live"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_status = {
        "multimodal_detector": multimodal_detector is not None,
        "image_detector": image_detector is not None,
        "video_detector": video_detector is not None,
        "audio_detector": audio_detector is not None,
        "dataset_manager": dataset_manager is not None
    }
    
    all_healthy = all(models_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models": models_status,
        "uptime": "active"
    }

@app.post("/detect/image")
async def detect_image_deepfake(file: UploadFile = File(...)):
    """
    Detect deepfakes in uploaded image
    """
    if not image_detector:
        raise HTTPException(status_code=503, detail="Image detector not available")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / f"temp_image_{uuid.uuid4().hex}.{file.filename.split('.')[-1]}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform detection
        result = image_detector.detect(str(file_path))
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "analysis_type": "single_modality_image",
            "timestamp": datetime.now().isoformat()
        })
        
        # Cleanup
        os.unlink(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        if file_path.exists():
            os.unlink(file_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/video")
async def detect_video_deepfake(file: UploadFile = File(...)):
    """
    Detect deepfakes in uploaded video
    """
    if not video_detector:
        raise HTTPException(status_code=503, detail="Video detector not available")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / f"temp_video_{uuid.uuid4().hex}.{file.filename.split('.')[-1]}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform detection
        result = video_detector.detect(str(file_path))
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "analysis_type": "single_modality_video",
            "timestamp": datetime.now().isoformat()
        })
        
        # Cleanup
        os.unlink(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Video detection failed: {e}")
        if file_path.exists():
            os.unlink(file_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/audio")
async def detect_audio_deepfake(file: UploadFile = File(...)):
    """
    Detect deepfakes in uploaded audio
    """
    if not audio_detector:
        raise HTTPException(status_code=503, detail="Audio detector not available")
    
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / f"temp_audio_{uuid.uuid4().hex}.{file.filename.split('.')[-1]}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform detection
        result = audio_detector.detect(str(file_path))
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "analysis_type": "single_modality_audio",
            "timestamp": datetime.now().isoformat()
        })
        
        # Cleanup
        os.unlink(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Audio detection failed: {e}")
        if file_path.exists():
            os.unlink(file_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/multimodal")
async def detect_multimodal_deepfake(
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Perform multi-modal deepfake detection
    """
    if not multimodal_detector:
        raise HTTPException(status_code=503, detail="Multi-modal detector not available")
    
    if not any([image, video, audio]):
        raise HTTPException(status_code=400, detail="At least one file must be provided")
    
    temp_files = {}
    
    try:
        # Save uploaded files temporarily
        if image and image.content_type.startswith('image/'):
            image_path = UPLOAD_DIR / f"temp_image_{uuid.uuid4().hex}.{image.filename.split('.')[-1]}"
            with open(image_path, "wb") as buffer:
                buffer.write(await image.read())
            temp_files['image'] = str(image_path)
        
        if video and video.content_type.startswith('video/'):
            video_path = UPLOAD_DIR / f"temp_video_{uuid.uuid4().hex}.{video.filename.split('.')[-1]}"
            with open(video_path, "wb") as buffer:
                buffer.write(await video.read())
            temp_files['video'] = str(video_path)
        
        if audio and audio.content_type.startswith('audio/'):
            audio_path = UPLOAD_DIR / f"temp_audio_{uuid.uuid4().hex}.{audio.filename.split('.')[-1]}"
            with open(audio_path, "wb") as buffer:
                buffer.write(await audio.read())
            temp_files['audio'] = str(audio_path)
        
        # Perform multi-modal detection
        result = multimodal_detector.detect_multi_modal(temp_files)
        
        # Add metadata
        result.update({
            "uploaded_files": {
                "image": image.filename if image else None,
                "video": video.filename if video else None,
                "audio": audio.filename if audio else None
            },
            "analysis_type": "multi_modal",
            "timestamp": datetime.now().isoformat(),
            "processing_time": "simulated", # Would be actual time in production
            "modalities_analyzed": len(temp_files)
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Multi-modal detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        for file_path in temp_files.values():
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup {file_path}: {cleanup_error}")

@app.get("/demo/samples")
async def get_demo_samples():
    """
    Get available demo samples for testing
    """
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not available")
    
    try:
        # Initialize dataset if not already done
        dataset_info = dataset_manager.initialize_complete_dataset()
        
        return JSONResponse(content={
            "status": "success",
            "dataset_info": dataset_info,
            "available_samples": dataset_manager.list_available_samples(),
            "demo_ready": True
        })
        
    except Exception as e:
        logger.error(f"Failed to get demo samples: {e}")
        raise HTTPException(status_code=500, detail=f"Demo samples unavailable: {str(e)}")

@app.post("/demo/test-case/{test_case_id}")
async def run_demo_test_case(test_case_id: str):
    """
    Run a specific demo test case
    """
    if not dataset_manager or not multimodal_detector:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    try:
        # Get test case
        test_case = dataset_manager.get_sample_for_testing(test_case_id)
        if not test_case:
            raise HTTPException(status_code=404, detail=f"Test case {test_case_id} not found")
        
        # Simulate detection results (in production, would use actual files)
        simulated_result = {
            "test_case_id": test_case_id,
            "test_name": test_case["name"],
            "description": test_case["description"],
            "files_analyzed": test_case["files"],
            "prediction": test_case["expected_result"]["prediction"],
            "confidence": (test_case["expected_result"]["confidence_range"][0] + 
                         test_case["expected_result"]["confidence_range"][1]) / 2,
            "agreement_score": 0.85 if test_case["expected_result"]["agreement_expected"] == "high" else 0.65,
            "risk_level": test_case["expected_result"]["risk_level"],
            "test_focus": test_case["test_focus"],
            "analysis_type": "demo_test_case",
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
            "expected_vs_actual": {
                "expected": test_case["expected_result"],
                "actual": "simulated_match"
            }
        }
        
        return JSONResponse(content=simulated_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo test case failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test case execution failed: {str(e)}")

@app.get("/analytics/dashboard")
async def get_dashboard_analytics():
    """
    Get analytics data for the dashboard
    """
    try:
        # Simulate analytics data (in production, would come from database)
        analytics = {
            "overview": {
                "total_analyses": 1247,
                "real_detected": 892,
                "ai_generated_detected": 355,
                "accuracy_rate": 94.2,
                "last_updated": datetime.now().isoformat()
            },
            "modality_breakdown": {
                "image_analyses": 567,
                "video_analyses": 423,
                "audio_analyses": 257,
                "multimodal_analyses": 312
            },
            "recent_activity": [
                {
                    "timestamp": "2024-01-15T10:30:00",
                    "type": "multimodal",
                    "result": "AI-Generated",
                    "confidence": 89.3,
                    "modalities": ["image", "audio"]
                },
                {
                    "timestamp": "2024-01-15T10:25:00",
                    "type": "video",
                    "result": "Real",
                    "confidence": 92.1,
                    "modalities": ["video"]
                },
                {
                    "timestamp": "2024-01-15T10:20:00",
                    "type": "image",
                    "result": "AI-Generated",
                    "confidence": 87.5,
                    "modalities": ["image"]
                }
            ],
            "detection_trends": {
                "daily_detections": [45, 52, 38, 61, 48, 55, 42],
                "accuracy_trend": [93.1, 94.2, 93.8, 94.5, 94.2, 94.7, 94.2],
                "risk_distribution": {
                    "high": 23,
                    "medium": 45,
                    "low": 67
                }
            },
            "system_performance": {
                "average_processing_time": "2.3s",
                "model_confidence": "High",
                "system_load": "Normal",
                "uptime": "99.8%"
            }
        }
        
        return JSONResponse(content=analytics)
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics unavailable")

@app.post("/analyze/start")
async def start_analysis(background_tasks: BackgroundTasks):
    """
    Start a background analysis task
    """
    analysis_id = str(uuid.uuid4())
    
    # Store analysis info
    active_analyses[analysis_id] = {
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Add background task (simulated)
    background_tasks.add_task(simulate_analysis, analysis_id)
    
    return JSONResponse(content={
        "analysis_id": analysis_id,
        "status": "started",
        "message": "Analysis started successfully"
    })

async def simulate_analysis(analysis_id: str):
    """
    Simulate a long-running analysis task
    """
    try:
        for progress in range(0, 101, 10):
            active_analyses[analysis_id]["progress"] = progress
            await asyncio.sleep(0.5)  # Simulate processing time
        
        # Complete analysis
        active_analyses[analysis_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress": 100,
            "result": {
                "prediction": "AI-Generated",
                "confidence": 91.5,
                "details": "Simulated analysis result"
            }
        })
        
    except Exception as e:
        active_analyses[analysis_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

@app.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Get the status of a running analysis
    """
    if analysis_id not in active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return JSONResponse(content=active_analyses[analysis_id])

@app.get("/models/info")
async def get_model_info():
    """
    Get information about loaded models
    """
    model_info = {
        "multimodal_detector": {
            "status": "loaded" if multimodal_detector else "not_loaded",
            "description": "Multi-modal fusion system for comprehensive analysis",
            "capabilities": ["image+video+audio", "adaptive_fusion", "confidence_weighting"]
        },
        "image_detector": {
            "status": "loaded" if image_detector else "not_loaded",
            "description": "EfficientNet-B4 based image deepfake detector",
            "capabilities": ["facial_analysis", "attention_visualization", "artifact_detection"]
        },
        "video_detector": {
            "status": "loaded" if video_detector else "not_loaded", 
            "description": "3D CNN + LSTM temporal video analyzer",
            "capabilities": ["temporal_consistency", "frame_analysis", "motion_patterns"]
        },
        "audio_detector": {
            "status": "loaded" if audio_detector else "not_loaded",
            "description": "Transformer-based synthetic speech detector",
            "capabilities": ["spectral_analysis", "prosody_detection", "voice_stress"]
        }
    }
    
    return JSONResponse(content=model_info)

# Mount static files for serving visualizations
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
