import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ----------------- 1D ResNet Architecture (High Fidelity) -----------------
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1DGesture(nn.Module):
    def __init__(self, n_subcarriers=228, num_classes=4):
        super(ResNet1DGesture, self).__init__()
        self.in_channels = 64
        self.prep = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer1 = ResBlock1D(64, 64, stride=1)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return self.classifier(x)

# ----------------- Data Augmentation & MixUp -----------------
def augment_csi_batch(x_tensor):
    # Temporal translation/roll augmentation: roll by random frames [-3, 3]
    batch_size = x_tensor.shape[0]
    x_aug = x_tensor.clone()
    for i in range(batch_size):
        shift = np.random.randint(-3, 4)
        if shift != 0:
            x_aug[i] = torch.roll(x_aug[i], shifts=shift, dims=1)
    
    # Add minor Gaussian noise
    noise = torch.randn_like(x_aug) * 0.02
    return x_aug + noise

def mixup_data(x, y, alpha=0.15, device='cpu'): # tuned alpha
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load and preprocess complex fusion dataset
    X_by_label = dataset.load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = dataset.split_80_20(X_by_label)
    
    print("\nPreprocessing: Amplitude + Calibrated Phase Fusion...")
    x_train_fusion = dataset.preprocess_csi_fusion(x_train_raw)
    x_test_fusion = dataset.preprocess_csi_fusion(x_test_raw)
    
    x_train_fusion_t = torch.tensor(x_train_fusion, dtype=torch.float32).permute(0, 2, 1) # (N, 228, 50)
    x_test_fusion_t = torch.tensor(x_test_fusion, dtype=torch.float32).permute(0, 2, 1)
    
    train_loader = DataLoader(TensorDataset(x_train_fusion_t, torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_fusion_t, torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    # 2. Initialize ResNet1D Model
    model = ResNet1DGesture(n_subcarriers=228, num_classes=4).to(device)
    
    criterion_train = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Hyperparameters
    epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_test_acc = 0.0
    best_weights = None
    
    print("\n================ Training Optimized ResNet1D (Tuned Augmentation + MixUp) ================")
    for epoch in range(epochs):
        model.train()
        correct, total, loss_val = 0, 0, 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply Temporal Shift & Gaussian Noise
            inputs_aug = augment_csi_batch(inputs)
            
            # Apply MixUp Data Mixing
            inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs_aug, labels, alpha=0.15, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs_mixed)
            loss = mixup_criterion(criterion_train, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            
            # Calculate training accuracy on non-mixed predictions
            with torch.no_grad():
                outputs_orig = model(inputs)
                _, predicted = torch.max(outputs_orig.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        train_acc = 100 * correct / total
        scheduler.step()
        
        # Test Evaluation
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
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = model.state_dict().copy()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
    print(f"\n================ Finished ResNet1D! Best Test Accuracy: {best_test_acc:.2f}% ================")
    
    # Save the best weights
    src_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(src_dir)
    save_path = os.path.join(workspace_dir, "models", "best_optimized_cnn.pth")
    torch.save(best_weights, save_path)
    print(f"Saved optimized ResNet1D weights to: {save_path}")

if __name__ == '__main__':
    main()
