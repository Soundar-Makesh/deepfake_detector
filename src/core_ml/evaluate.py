import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Import your optimized engine
from src.api.engine import ml_engine

def run_evaluation(data_dir="data/raw", meta_file="metadata.json", output_dir="results"):
    print("\n[+] Initializing SENTRY AI Evaluation Protocol...")
    
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, meta_file)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    y_true = []
    y_pred = []
    y_probs = []
    processing_times = []
    failed_videos = 0

    print(f"[+] Found {len(meta)} videos in metadata. Starting inference...\n")

    # Using tqdm for a beautiful progress bar
    for video_file, data in tqdm(meta.items(), desc="Evaluating Models", unit="vid"):
        video_path = os.path.join(data_dir, video_file)
        
        if not os.path.exists(video_path):
            continue

        true_label = 1 if data['label'] == 'FAKE' else 0
        
        # Track Latency for the "Real-Time" requirement proof
        start_time = time.time()
        result = ml_engine.analyze(video_path)
        inference_time = time.time() - start_time
        
        if result['prediction'] == "ERROR":
            failed_videos += 1
            continue

        pred_label = 1 if result['prediction'] == 'FAKE' else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_probs.append(result['raw_probability'])
        processing_times.append(inference_time)

    # --- METRICS CALCULATION ---
    print("\n\n" + "="*50)
    print(" SENTRY AI - FORMAL EVALUATION REPORT")
    print("="*50)
    
    avg_fps = 1.0 / np.mean(processing_times)
    print(f"Total Videos Processed : {len(y_true)}")
    print(f"Pipeline Failures      : {failed_videos} (Face not detected)")
    print(f"Average Latency        : {np.mean(processing_times):.3f} seconds/video")
    print(f"Estimated Throughput   : {avg_fps:.1f} FPS")
    print("-" * 50)
    
    # The Scikit-Learn Classification Report
    report = classification_report(y_true, y_pred, target_names=["REAL (0)", "FAKE (1)"])
    print(report)

    # Save report to text file
    with open(f"{output_dir}/evaluation_report.txt", "w") as f:
        f.write("SENTRY AI EVALUATION REPORT\n")
        f.write(f"Average Latency: {np.mean(processing_times):.3f}s\n\n")
        f.write(report)

    # --- VISUALIZATION 1: CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Predicted REAL", "Predicted FAKE"], 
                yticklabels=["Actual REAL", "Actual FAKE"])
    plt.title('Deepfake Detection Confusion Matrix', pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()

    # --- VISUALIZATION 2: ROC CURVE ---
    # The ROC curve proves how well the model separates the two classes
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', pad=20, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
    plt.close()

    print(f"\n[+] Success! Visualizations saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    run_evaluation()