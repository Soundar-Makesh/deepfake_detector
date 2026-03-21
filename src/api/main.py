import os
import shutil
import uuid
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
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

app.mount("/", StaticFiles(directory="static", html=True), name="static")