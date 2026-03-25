import os
import shutil
import uuid
import torch
import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from src.api.engine import predict_video

app = FastAPI(title="Deepfake Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/api/health")
async def health_check():
    """Verify GPU status and model readiness"""
    cuda_available = torch.cuda.is_available()
    return {
        "status": "online",
        "device": "RTX 3050" if cuda_available else "CPU (Fallback)",
        "cuda": cuda_available
    }

@app.get("/api/info")
async def api_info():
    """Deployment API endpoint providing system metadata (R9)"""
    return {
        "api_version": "1.0",
        "model_architecture": "DeepfakeHybridModel",
        "deployment_status": "Production",
        "environment": "CUDA" if torch.cuda.is_available() else "CPU",
        "supported_formats": ["mp4", "avi", "mov"]
    }

@app.post("/api/predict")
async def run_prediction(video: UploadFile = File(...)):
    if not video.filename or not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4/AVI.")
        
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        result = await run_in_threadpool(predict_video, file_path)
        
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Internal Error] {str(e)}") 
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")
        
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Cleanup Error] Could not delete {file_path}: {e}")

@app.post("/api/feedback")
async def submit_feedback(true_label: str = Form(...), video: UploadFile = File(...)):
    """Continuous Dataset Update endpoint (R10)"""
    if true_label not in ["REAL", "FAKE"]:
        raise HTTPException(status_code=400, detail="true_label must be REAL or FAKE")
        
    dataset_dir = os.path.join("data", "raw", true_label)
    os.makedirs(dataset_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{video.filename}"
    file_path = os.path.join(dataset_dir, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        return {"status": "success", "message": f"Dataset updated with {true_label} video.", "dataset_path": file_path}
    except Exception as e:
        print(f"[Feedback Error] {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save dataset update.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")