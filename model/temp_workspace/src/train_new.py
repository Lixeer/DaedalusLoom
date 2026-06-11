# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Adjust path to import dataset and models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset
import models

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------- Transformer Architecture -----------------
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
    def __init__(self, input_dim=228, d_model=128, nhead=4, num_layers=2, num_classes=4):
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
        # input shape: (batch, channels, time) = (batch, 228, 50)
        x = x.permute(0, 2, 1) # -> (batch, 50, 228)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling over temporal axis
        return self.classifier(x)

# ----------------- Training Pipeline -----------------
def train_configuration(config_name, model, train_loader, test_loader, epochs, device):
    print(f"\n================ Training Configuration: {config_name} ================")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
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
        
        # Evaluation
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
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
    print(f"Finished {config_name}! Best Test Accuracy: {best_test_acc:.2f}%")
    return best_test_acc, history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Establish organized directories
    src_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(src_dir)
    plots_dir = os.path.join(workspace_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load raw dataset using dataset.py
    X_by_label = dataset.load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = dataset.split_80_20(X_by_label)
    
    # 2. Preprocess: Fused Amplitude + Calibrated Phase
    print("\n--- Preprocessing: Amplitude + Calibrated Phase Fusion ---")
    x_train_fusion = dataset.preprocess_csi_fusion(x_train_raw)
    x_test_fusion = dataset.preprocess_csi_fusion(x_test_raw)
    
    # Permute to (N, channels, time) = (N, 228, 50)
    x_train_t = torch.tensor(x_train_fusion, dtype=torch.float32).permute(0, 2, 1)
    x_test_t = torch.tensor(x_test_fusion, dtype=torch.float32).permute(0, 2, 1)
    
    train_loader = DataLoader(TensorDataset(x_train_t, torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    epochs = 20
    all_results = {}
    
    # =========================================================================
    # Config 1: Simple MLP (Fused Amp + Phase)
    # =========================================================================
    model_mlp = models.SimpleMLP(input_dim=11400, num_classes=4).to(device)
    best_acc_mlp, history_mlp = train_configuration(
        "MLP (Amp + Phase)", model_mlp, train_loader, test_loader, epochs=epochs, device=device
    )
    all_results["MLP"] = (best_acc_mlp, history_mlp)
    
    # =========================================================================
    # Config 2: CNN1D (Fused Amp + Phase)
    # =========================================================================
    model_cnn = models.Advanced1DCNN(n_subcarriers=228, num_classes=4).to(device)
    best_acc_cnn, history_cnn = train_configuration(
        "CNN1D (Amp + Phase)", model_cnn, train_loader, test_loader, epochs=epochs, device=device
    )
    all_results["CNN1D"] = (best_acc_cnn, history_cnn)
    
    # =========================================================================
    # Config 3: LSTM (Fused Amp + Phase)
    # =========================================================================
    model_lstm = models.LSTMGestureClassifier(input_dim=228, num_classes=4).to(device)
    best_acc_lstm, history_lstm = train_configuration(
        "LSTM (Amp + Phase)", model_lstm, train_loader, test_loader, epochs=epochs, device=device
    )
    all_results["LSTM"] = (best_acc_lstm, history_lstm)
    
    # =========================================================================
    # Config 4: Transformer (Fused Amp + Phase)
    # =========================================================================
    model_transformer = CSITransformer(input_dim=228, d_model=128, nhead=4, num_layers=2, num_classes=4).to(device)
    best_acc_trans, history_trans = train_configuration(
        "Transformer (Amp + Phase)", model_transformer, train_loader, test_loader, epochs=epochs, device=device
    )
    all_results["Transformer"] = (best_acc_trans, history_trans)
    
    # 3. Generate Evaluation Plot (Test Accuracy)
    plt.figure(figsize=(10, 6))
    for name, (best_acc, history) in all_results.items():
        plt.plot(range(1, epochs + 1), history['test_acc'], marker='o', label=f"{name} (Best: {best_acc:.2f}%)")
        
    plt.title('Test Accuracy Comparison across Architectures (Amplitude + Phase Fusion)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    comparison_filename = "comparison_new.png"
    plt.savefig(os.path.join(plots_dir, comparison_filename))
    plt.close()
    print(f"\nSaved overall comparison plot to {comparison_filename} inside plots/ directory.")
    
    # 4. Print final summary table
    print("\n================ Final Performance Summary ================")
    print(f"{'Architecture':15s} | {'Best Test Accuracy (%)':22s}")
    print("-" * 42)
    for name, (acc, _) in all_results.items():
        print(f"{name:15s} | {acc:22.2f}%")

if __name__ == '__main__':
    main()
