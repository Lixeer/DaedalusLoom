import os
import numpy as np

#LABEL_MAP = {'bow': 0, 'boxing': 1, 'draw_o': 2, 'stand': 3}
LABEL_MAP = {'cut': 0, 'grip': 1, 'draw_o': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_dataset(dataset_dir="dataset/dataset_2026_6_10"):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    
    npz_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npz')])
    X_dict = {}
    
    print(f"Loading dataset from {dataset_dir}...")
    for filename in npz_files:
        file_path = os.path.join(dataset_dir, filename)
        
        # Robust label extraction: find index before the 8-digit date string starting with '202'
        parts = filename.split('_')
        date_idx = -1
        for idx, part in enumerate(parts):
            if len(part) == 8 and part.isdigit() and part.startswith('202'):
                date_idx = idx
                break
        
        if date_idx != -1:
            label_name = '_'.join(parts[:date_idx])
        else:
            label_name = parts[0]
            
        if label_name not in LABEL_MAP:
            print(f"  Warning: Skipping unknown label prefix '{label_name}' in file '{filename}'")
            continue
            
        label_idx = LABEL_MAP[label_name]
        data = np.load(file_path, allow_pickle=True)['dataset']
        
        if label_idx not in X_dict:
            X_dict[label_idx] = []
        X_dict[label_idx].append(data)
        
    X_by_label = {}
    for idx in X_dict:
        X_by_label[idx] = np.concatenate(X_dict[idx], axis=0)
        print(f"  Class '{INV_LABEL_MAP[idx]}': {X_by_label[idx].shape[0]} samples")
        
    return X_by_label

def split_80_20(X_by_label):
    x_train_list, x_test_list = [], []
    y_train_list, y_test_list = [], []
    
    print("\nSplitting dataset (80% train, 20% test sequentially)...")
    for idx in sorted(X_by_label.keys()):
        X = X_by_label[idx]
        n_samples = len(X)
        split_point = int(n_samples * 0.3)
        
        x_train_list.append(X[:split_point])
        x_test_list.append(X[split_point:])
        y_train_list.append(np.full(split_point, idx, dtype=np.int64))
        y_test_list.append(np.full(n_samples - split_point, idx, dtype=np.int64))
        print(f"  Class '{INV_LABEL_MAP[idx]}': Train={split_point}, Test={n_samples - split_point}")
        
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    return x_train, x_test, y_train, y_test

def preprocess_csi_fusion(X_complex, eps=2.0):
    N = len(X_complex)
    X_amp = np.abs(X_complex)
    X_phase = np.angle(X_complex)
    
    # Vectorized Linear Phase Calibration to remove CFO and SFO phase noise
    unwrapped = np.unwrap(X_phase, axis=2)
    x = np.arange(114)
    x_mean = x.mean()
    x_dev = x - x_mean
    D = np.sum(x_dev**2)
    
    Y_mean = unwrapped.mean(axis=2, keepdims=True)
    y_dev = unwrapped - Y_mean
    a = np.sum(y_dev * x_dev, axis=2, keepdims=True) / D
    X_phase_cal = y_dev - a * x_dev
    
    # Apply SR-Std to both amplitude and calibrated phase
    X_amp_norm = np.zeros_like(X_amp)
    X_phase_norm = np.zeros_like(X_phase_cal)
    
    for i in range(N):
        mean_amp = X_amp[i].mean(axis=0, keepdims=True)
        std_amp = X_amp[i].std(axis=0, keepdims=True)
        X_amp_norm[i] = (X_amp[i] - mean_amp) / (std_amp + eps)
        
        mean_phase = X_phase_cal[i].mean(axis=0, keepdims=True)
        std_phase = X_phase_cal[i].std(axis=0, keepdims=True)
        X_phase_norm[i] = (X_phase_cal[i] - mean_phase) / (std_phase + eps)
        
    # Concatenate amplitude and phase features along the subcarrier axis (50, 114) + (50, 114) -> (50, 228)
    X_combined = np.concatenate([X_amp_norm, X_phase_norm], axis=2)
    return X_combined

def preprocess_csi_amp_only(X_complex, eps=2.0):
    N = len(X_complex)
    X_amp = np.abs(X_complex)
    X_amp_norm = np.zeros_like(X_amp)
    
    for i in range(N):
        mean_amp = X_amp[i].mean(axis=0, keepdims=True)
        std_amp = X_amp[i].std(axis=0, keepdims=True)
        X_amp_norm[i] = (X_amp[i] - mean_amp) / (std_amp + eps)
        
    return X_amp_norm
