import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from src.core_ml.preprocess import process_video
from src.core_ml.model import DeepfakeHybridModel

class DFDCDataset(Dataset):
    def __init__(self, data_dir, num_frames=10):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.video_files = []
        self.labels = []
        
        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"CRITICAL: metadata.json not found in {data_dir}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        print("Indexing dataset...")
        for video_name, info in metadata.items():
            vid_path = os.path.join(data_dir, video_name)
            if os.path.exists(vid_path):
                self.video_files.append(vid_path)
                self.labels.append(1.0 if info['label'] == 'FAKE' else 0.0)
                
        print(f"Successfully indexed {len(self.video_files)} videos.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        attempts = 0 
        while attempts < 10:
            vid_path = self.video_files[idx]
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
            
            frames, fft_scores = process_video(vid_path, num_frames=self.num_frames)
            
            if frames is not None:
                return frames, fft_scores, label
                
            idx = (idx + 1) % len(self.video_files)
            attempts += 1
            
        raise RuntimeError("Too many corrupted videos encountered in a row.")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- INITIATING TRAINING ON: {device} ---")

    BATCH_SIZE = 4 
    ACCUMULATION_STEPS = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    from torch.utils.data import random_split
    full_dataset = DFDCDataset(data_dir="data/raw/", num_frames=5)
    if len(full_dataset) == 0:
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepfakeHybridModel(freeze_cnn=True).to(device)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scaler = torch.amp.GradScaler(device=device.type) if device.type == 'cuda' else None
    
    os.makedirs("src/models", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)
        
        for i, (frames, fft_scores, labels) in enumerate(loop):
            frames = frames.to(device, non_blocking=True)
            fft_scores = fft_scores.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(frames, fft_scores)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps
                loss = loss / ACCUMULATION_STEPS
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            running_loss += (loss.item() * ACCUMULATION_STEPS) * frames.size(0)
            predicted = (outputs > 0).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            current_loss = running_loss / total_samples
            current_acc = (correct_predictions / total_samples) * 100
            loop.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.inference_mode():
            for frames, fft_scores, labels in val_loader:
                frames = frames.to(device, non_blocking=True)
                fft_scores = fft_scores.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    outputs = model(frames, fft_scores)
                    v_loss = criterion(outputs, labels)
                    
                val_loss += v_loss.item() * frames.size(0)
                predicted = (outputs > 0).float()
                val_correct += (predicted == labels).sum().item()
                
        epoch_loss = val_loss / val_size
        val_acc = (val_correct / val_size) * 100
        print(f"Validation => Loss: {epoch_loss:.4f} | Acc: {val_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "src/models/deepfake_mvp.pth")
            print("\n[+] Model improved! Saved new weights to src/models/deepfake_mvp.pth")

    print("\n--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    train_model()