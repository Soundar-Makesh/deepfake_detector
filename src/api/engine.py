import torch
import cv2
import numpy as np
import os
import mediapipe as mp
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

class DeepfakeEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Booting up ViT Deepfake Detector on {self.device}...")
        
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()
        self.is_ready = True
        
        # Cache label mapping
        self.label_map = self.model.config.id2label
        print(f"[Engine] ViT model loaded. Labels: {self.label_map}")
        print(f"[Engine] Ready for inference on {self.device}.")
        
        # MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def _extract_face(self, frame_bgr):
        """Detect and crop face from a BGR frame. Returns PIL Image or None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return None
            
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame_bgr.shape
        
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
            
        face_crop = frame_rgb[y1:y2, x1:x2]
        return Image.fromarray(face_crop)

    def predict(self, video_path):
        if not self.is_ready:
            return {"error": "Model failed to load."}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5:
            cap.release()
            return {"error": "Video is too short for analysis."}

        # Sample 10 frames spread across the video
        sample_positions = [int(total_frames * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]]
        
        fake_scores = []
        
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
                
            face_img = self._extract_face(frame)
            if face_img is None:
                continue
            
            # Process with ViT
            inputs = self.processor(images=face_img, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Find the "Deepfake" class probability
            fake_prob = 0.0
            for idx, label in self.label_map.items():
                if "fake" in str(label).lower() or "deepfake" in str(label).lower():
                    fake_prob = probs[0][int(idx)].item()
                    break
            
            fake_scores.append(fake_prob)
        
        cap.release()
        
        if not fake_scores:
            return {"error": "Could not detect a clear face in any frame of the video."}
        
        # Diagnostic logging
        avg_fake = sum(fake_scores) / len(fake_scores)
        print(f"[Forensics] Per-frame fake scores: {[round(s, 3) for s in fake_scores]} | Avg: {avg_fake:.4f}")
        
        # Simple honest classification
        if avg_fake > 0.5:
            prediction = "FAKE"
            display_confidence = avg_fake
        else:
            prediction = "REAL"
            display_confidence = 1.0 - avg_fake
            
        return {
            "prediction": prediction,
            "confidence": round(display_confidence * 100, 2),
            "raw_probability": round(avg_fake, 4),
            "frames_analyzed": len(fake_scores)
        }

ml_engine = DeepfakeEngine()

def predict_video(video_path):
    return ml_engine.predict(video_path)