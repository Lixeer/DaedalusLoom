import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

LABEL_MAP = {'draw': 0, 'stand-up': 1, 'wave': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_dataset(dataset_dir="dataset/dataset_2026_6_23"):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    npz_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npz')])
    X_dict = {0: [], 1: [], 2: []}
    
    print(f"Loading dataset from {dataset_dir}...")
    for filename in sorted(npz_files):
        file_path = os.path.join(dataset_dir, filename)
        label_name = filename.split('_')[0]
        if label_name not in LABEL_MAP:
            continue
        label_idx = LABEL_MAP[label_name]
        data = np.load(file_path, allow_pickle=True)['dataset']
        X_dict[label_idx].append(data)
        
    X_by_label = {}
    for idx in X_dict:
        X_by_label[idx] = np.concatenate(X_dict[idx], axis=0)
        print(f"  Class '{INV_LABEL_MAP[idx]}': {X_by_label[idx].shape[0]} samples")
        
    return X_by_label

def split_50_50(X_by_label):
    x_train_list, x_test_list = [], []
    y_train_list, y_test_list = [], []
    
    print("\nSplitting dataset (50% train, 50% test sequentially)...")
    for idx in sorted(X_by_label.keys()):
        X = X_by_label[idx]
        n_samples = len(X)
        split_point = int(n_samples * 0.5)
        
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

def preprocess_sr_std(X, eps=2.0):
    # Subcarrier-wise Regularized Standardization (SR-Std)
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        mean = X[i].mean(axis=0, keepdims=True)
        std = X[i].std(axis=0, keepdims=True)
        X_norm[i] = (X[i] - mean) / (std + eps)
    return X_norm

class FCN(nn.Module):
    def __init__(self, input_dim=114, num_classes=3):
        super(FCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # input x shape: (batch, input_dim, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        return self.fc(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load raw dataset
    X_by_label = load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = split_50_50(X_by_label)
    
    # Apply Preprocessing (SR-Std)
    print("\nApplying Subcarrier-wise Regularized Standardization (SR-Std)...")
    x_train = preprocess_sr_std(x_train_raw, eps=2.0)
    x_test = preprocess_sr_std(x_test_raw, eps=2.0)
    
    # Convert to Tensor and permute to (batch, subcarriers, time) for 1D convolution
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 2, 1)
    
    train_dataset = TensorDataset(x_train_tensor, torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(x_test_tensor, torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = FCN(input_dim=114, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    epochs = 30
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0.0
    
    print("\n=== Starting Training (FCN + SR-Std + 50/50 Sequential Split) ===")
    for epoch in range(epochs):
        model.train()
        correct, total, loss_val = 0, 0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        scheduler.step()
        
        # Test evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_acc = 100 * test_correct / test_total
        avg_loss = loss_val / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(script_dir, "best_fcn.pth"))
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
    print(f"\nTraining Complete! Best Test Accuracy: {best_test_acc:.2f}%")
    
    # Save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "fcn_history.png"))
    print(f"Saved training curves to '{os.path.join(script_dir, 'fcn_history.png')}'")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(script_dir, "best_fcn.pth"), map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Per-class accuracy
    print("\n=== Per-Class Accuracy Evaluation ===")
    for class_name, class_idx in LABEL_MAP.items():
        indices = np.where(all_labels == class_idx)[0]
        class_correct = np.sum(all_preds[indices] == class_idx)
        class_acc = 100 * class_correct / len(indices)
        print(f"  Class '{class_name}': Acc={class_acc:.2f}% ({class_correct}/{len(indices)})")

if __name__ == '__main__':
    main()
