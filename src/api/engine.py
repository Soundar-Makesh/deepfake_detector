import torch
import os
from .model import DeepfakeModel
from ..core_ml.preprocess import process_video

class DeepfakeEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Booting up ML architecture on {self.device}...")
        
        self.model = DeepfakeModel().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "deepfake_mvp.pth")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"[Engine] Weights loaded from {model_path}. Ready for inference.")
            self.is_ready = True
        else:
            print(f"[Engine] ERROR: Model weights not found at {model_path}")
            self.is_ready = False
            
        self.model.eval()

    def predict(self, video_path):
        if not self.is_ready:
            return {"error": "Model weights are missing. Please ensure deepfake_mvp.pth exists."}

        offsets = [0.15, 0.33, 0.50, 0.66, 0.85]
        all_probs = []
        
        for offset in offsets:
            frames, fft_scores = process_video(video_path, num_frames=10, start_percent=offset)
            if frames is None:
                continue
                
            frames = frames.unsqueeze(0).to(self.device, non_blocking=True)
            fft_scores = fft_scores.unsqueeze(0).to(self.device, non_blocking=True)
            
            with torch.inference_mode():
                with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    logit = self.model(frames, fft_scores)
                    prob = torch.sigmoid(logit).item()
                    all_probs.append(prob)

        if not all_probs:
            return {"error": "Could not detect a clear face anywhere in the video sequence."}
        
        avg_prob = sum(all_probs) / len(all_probs)
        print(f"[Forensics] Scans: {[round(p, 3) for p in all_probs]} | Avg: {avg_prob:.4f}")
        
        if avg_prob > 0.5:
            prediction = "FAKE"
            display_confidence = avg_prob
        else:
            prediction = "REAL"
            display_confidence = 1.0 - avg_prob
            
        return {
            "prediction": prediction,
            "confidence": round(display_confidence * 100, 2),
            "raw_probability": round(avg_prob, 4),
            "scanned_sequences": len(all_probs)
        }

ml_engine = DeepfakeEngine()

def predict_video(video_path):
    return ml_engine.predict(video_path)