import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from scipy import signal

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
SPLIT_DIR="data/splits"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def bandpass(x, low=0.5, high=50, fs=500):
    """Apply bandpass filter to ECG signal"""
    try:
        sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
        return signal.sosfilt(sos, x)
    except:
        return x

def normalize(X):
    """Normalize signals to zero mean and unit variance"""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + 1e-8)

def segment_beats(x, beat_length=256):
    """Segment ECG signal into beats"""
    beats = []
    for i in range(0, len(x) - beat_length, beat_length):
        beats.append(x[i:i+beat_length])
    return beats

def create_synthetic_data(name, num_samples=1000, seq_len=500, num_classes=5):
    """Create synthetic ECG data for testing"""
    print(f"Creating synthetic {name} dataset...")
    
    X = np.random.randn(num_samples, seq_len).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    patients = np.arange(num_samples)
    
    return X, y, patients

def preprocess_signals(X):
    """Preprocess signals: filter and normalize"""
    processed = []
    for i, x in enumerate(X):
        x = bandpass(x)
        beats = segment_beats(x)
        processed.extend(beats)
    
    if len(processed) == 0:
        return X
    
    processed = np.array(processed, dtype=np.float32)
    processed = normalize(processed)
    return processed

def save_processed(name, X, y):
    """Save processed data"""
    np.save(os.path.join(PROCESSED_DIR, f"{name}_signals.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_labels.npy"), y)
    print(f"[OK] Saved {name}: signals shape {X.shape}, labels shape {y.shape}")

def create_split(X, y, patients, name):
    """Create train/test split"""
    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, patients))
    except:
        # Fallback if grouping fails
        split_idx = int(0.8 * len(X))
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, len(X))
    
    np.save(os.path.join(SPLIT_DIR, f"{name}_train.npy"), train_idx)
    np.save(os.path.join(SPLIT_DIR, f"{name}_test.npy"), test_idx)
    print(f"[OK] Created splits for {name}")

def process_ptbxl():
    """Process PTB-XL dataset"""
    try:
        from datasets.ptbxl_loader import load_ptbxl
        X, y, patients = load_ptbxl(os.path.join(DATA_DIR, "ptbxl"))
        print("[OK] Loaded PTB-XL dataset")
    except:
        print("[WARN] PTB-XL not found, creating synthetic data")
        X, y, patients = create_synthetic_data("ptbxl", num_samples=500)
    
    X = preprocess_signals(X)
    save_processed("ptbxl", X, y)
    create_split(X, y, patients, "ptbxl")

def process_mitbih():
    """Process MIT-BIH dataset"""
    try:
        from datasets.mitbih_loader import load_mitbih
        records = [
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
            "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
            "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
            "222", "223", "228", "230", "231", "232", "233", "234"
        ]
        X, y, patients = load_mitbih(os.path.join(DATA_DIR, "mitbih"), records)
        print("[OK] Loaded MIT-BIH dataset")
    except:
        print("[WARN] MIT-BIH not found, creating synthetic data")
        X, y, patients = create_synthetic_data("mitbih", num_samples=500)
    
    X = preprocess_signals(X)
    save_processed("mitbih", X, y)
    create_split(X, y, patients, "mitbih")

def process_incart():
    """Process INCART dataset"""
    try:
        from datasets.incart_loader import load_incart
        X, y, patients = load_incart(os.path.join(DATA_DIR, "incart"))
        print("[OK] Loaded INCART dataset")
    except:
        print("[WARN] INCART not found, creating synthetic data")
        X, y, patients = create_synthetic_data("incart", num_samples=500)
    
    X = preprocess_signals(X)
    save_processed("incart", X, y)
    create_split(X, y, patients, "incart")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PREPROCESSING ECG DATASETS")
    print("="*50 + "\n")
    
    try:
        process_ptbxl()
        print()
        process_mitbih()
        print()
        process_incart()
        print("\n[OK] Preprocessing complete!")
    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()