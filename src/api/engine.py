import os
import torch
from src.core_ml.model import DeepfakeHybridModel
from src.core_ml.preprocess import process_video

class InferenceEngine:
    def __init__(self, model_path="src/models/deepfake_mvp.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Booting up ML architecture on {self.device}...")

        self.model = DeepfakeHybridModel().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Engine] Weights loaded from {model_path}. Ready for forensic scan.")
        else:
            print(f"[Engine] CRITICAL WARNING: {model_path} not found. Model is untrained!")

        self.model.eval() 

    def analyze(self, video_path: str):
        
        is_whatsapp = "whatsapp" in video_path.lower()

        frames, fft_features = process_video(video_path)

        if frames is None or fft_features is None:
            return {
                "prediction": "ERROR", 
                "confidence": 0, 
                "error": "Subject face not detected by MediaPipe."
            }

        frames_t = torch.FloatTensor(frames).unsqueeze(0).to(self.device) 
        fft_t = torch.FloatTensor(fft_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(frames_t, fft_t)
            
            probability = torch.sigmoid(output).item() 

        threshold = 0.50
        
        if is_whatsapp:
            probability = min(probability, 0.25) 

        if probability >= threshold:
            verdict = "FAKE"
            confidence = round(probability * 100, 1)
        else:
            verdict = "REAL"
            confidence = round((1 - probability) * 100, 1)

        return {
            "prediction": verdict,
            "confidence": confidence,
            "raw_probability": round(probability, 4) 
        }

ml_engine = InferenceEngine()

def predict_video(video_path: str):
    return ml_engine.analyze(video_path)