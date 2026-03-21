import cv2
import mediapipe as mp
import numpy as np
import torch

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_fft_feature(gray_face, bins=8):
    f = np.fft.fft2(gray_face)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.hypot(x, y).astype(int)
    
    radial_mean = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
    length = min(len(radial_mean), bins)
    profile = np.zeros(bins, dtype=np.float32)
    profile[:length] = radial_mean[:length]
    return profile

def process_video(video_path, num_frames=5, img_size=224, start_percent=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        cap.release()
        return None, None

    frames_data = []
    fft_data = []
    
    # Enable scanning at arbitrary offsets (e.g. 25%, 50%, 75%)
    start_frame = int(total_frames * start_percent) - (num_frames // 2)
    start_frame = max(0, min(start_frame, total_frames - num_frames))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: 
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            
            x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
            w, h = int(bbox.width * iw), int(bbox.height * ih)
            
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            x_min = max(0, x - pad_x)
            y_min = max(0, y - pad_y)
            x_max = min(iw, x + w + pad_x)
            y_max = min(ih, y + h + pad_y)
            
            face_crop = rgb_frame[y_min:y_max, x_min:x_max]
            
            if face_crop.size == 0: 
                continue
            
            face_resized = cv2.resize(face_crop, (img_size, img_size))
            
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            fft_score = get_fft_feature(gray_face)
            
            face_tensor = face_resized.astype(np.float32) / 255.0
            face_tensor = np.transpose(face_tensor, (2, 0, 1))
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            face_tensor = (face_tensor - mean) / std
            
            frames_data.append(face_tensor)
            fft_data.append(fft_score)
            
        if len(frames_data) == num_frames:
            break
            
    cap.release()
    
    if len(frames_data) < num_frames:
        return None, None 
        
    return torch.tensor(np.array(frames_data)), torch.tensor(np.array(fft_data), dtype=torch.float32)