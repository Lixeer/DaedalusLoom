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

# ----------------- Data Loading & Preprocessing -----------------
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
        split_point = int(n_samples * 0.8)
        
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

# ----------------- Model Architectures -----------------

# 1. Fully Convolutional Network (FCN)
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        return self.fc(x)

# 2. 1D Convolutional Neural Network (CNN)
class Advanced1DCNN(nn.Module):
    def __init__(self, n_subcarriers=114, num_classes=3):
        super(Advanced1DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# 3. Multi-Layer Perceptron (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5700, num_classes=3): # 114 * 50 = 5700
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 4. Long Short-Term Memory (LSTM)
class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_dim=114, hidden_dim=128, num_layers=2, num_classes=3):
        super(LSTMGestureClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1) # -> (batch, 50, 114)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) # Bidirectional last hidden states
        return self.classifier(last_hidden)

# 5. Transformer Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CSITransformer(nn.Module):
    def __init__(self, input_dim=114, d_model=128, nhead=4, num_layers=2, num_classes=3):
        super(CSITransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # -> (batch, 50, 114)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling over temporal axis
        return self.classifier(x)

# ----------------- Training Pipeline -----------------
def train_model(model_name, model, train_loader, test_loader, epochs, device):
    print(f"\n--- Training {model_name} ---")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_weights = None
    
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
            best_weights = model.state_dict().copy()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
    print(f"Finished {model_name}! Best Test Accuracy: {best_test_acc:.2f}%")
    return best_test_acc, best_weights, history

# ----------------- Main Execution -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load and Split Dataset (strictly sequentially 50/50, root directory only)
    X_by_label = load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = split_50_50(X_by_label)
    
    # 2. Preprocess (SR-Std)
    print("\nApplying Subcarrier-wise Regularized Standardization (SR-Std)...")
    x_train = preprocess_sr_std(x_train_raw, eps=2.0)
    x_test = preprocess_sr_std(x_test_raw, eps=2.0)
    
    # 3. Convert to Tensors and reshape to (batch, subcarriers, time)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 2, 1)
    
    train_dataset = TensorDataset(x_train_tensor, torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(x_test_tensor, torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define models to train
    models_to_train = {
        'FCN': (FCN(input_dim=114, num_classes=3), 'best_fcn.pth'),
        'CNN1D': (Advanced1DCNN(n_subcarriers=114, num_classes=3), 'best_cnn1d.pth'),
        'MLP': (SimpleMLP(input_dim=5700, num_classes=3), 'best_mlp.pth'),
        'LSTM': (LSTMGestureClassifier(input_dim=114, hidden_dim=128, num_layers=2, num_classes=3), 'best_lstm.pth'),
        'Transformer': (CSITransformer(input_dim=114, d_model=128, nhead=4, num_layers=2, num_classes=3), 'best_transformer.pth')
    }
    
    epochs = 30
    all_histories = {}
    best_accuracies = {}
    
    # Train each model
    for name, (model, weight_file) in models_to_train.items():
        model = model.to(device)
        best_acc, weights, history = train_model(name, model, train_loader, test_loader, epochs, device)
        best_accuracies[name] = best_acc
        all_histories[name] = history
        
        # Save weights
        torch.save(weights, os.path.join(script_dir, weight_file))
        print(f"Saved {name} weights to {weight_file}")
        
        # Save individual model plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title(f'{name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['test_acc'], label='Test Acc')
        plt.title(f'{name} Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plot_filename = f"{name.lower()}_history.png"
        plt.savefig(os.path.join(script_dir, plot_filename))
        plt.close()
        print(f"Saved {name} curves to {plot_filename}")

        # Final per-class evaluation for the best model state
        model.load_state_dict(weights)
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
        
        print(f"\n=== Per-Class Accuracy Evaluation for {name} ===")
        for class_name, class_idx in LABEL_MAP.items():
            indices = np.where(all_labels == class_idx)[0]
            class_correct = np.sum(all_preds[indices] == class_idx)
            class_acc = 100 * class_correct / len(indices)
            print(f"  Class '{class_name}': Acc={class_acc:.2f}% ({class_correct}/{len(indices)})")
            
    # 4. Generate Joint Comparison Plot
    plt.figure(figsize=(10, 6))
    for name, history in all_histories.items():
        plt.plot(range(1, epochs + 1), history['test_acc'], label=f"{name} (Best: {best_accuracies[name]:.2f}%)")
    plt.title('Test Accuracy Comparison across Architectures (SR-Std + 50/50 Split)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    comparison_filename = "model_comparison.png"
    plt.savefig(os.path.join(script_dir, comparison_filename))
    plt.close()
    print(f"\nSaved overall comparison plot to {comparison_filename}")
    
    # 5. Print final summary table
    print("\n================ Final Performance Summary ================")
    print(f"{'Model':15s} | {'Best Test Accuracy (%)':22s}")
    print("-" * 43)
    for name, acc in best_accuracies.items():
        print(f"{name:15s} | {acc:22.2f}%")

if __name__ == '__main__':
    main()
