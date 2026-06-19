import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Adjust path to import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset
import models

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

def train_configuration(config_name, model, train_loader, test_loader, epochs, device):
    print(f"\n================ Training Configuration: {config_name} ================")
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
            best_weights = model.state_dict().copy()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
    print(f"Finished {config_name}! Best Test Accuracy: {best_test_acc:.2f}%")
    return best_test_acc, best_weights, history

def evaluate_best_model(config_name, model, weights, test_loader, device):
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
    
    print(f"\n=== Per-Class Accuracy Evaluation for {config_name} ===")
    for class_name, class_idx in dataset.LABEL_MAP.items():
        indices = np.where(all_labels == class_idx)[0]
        if len(indices) > 0:
            class_correct = np.sum(all_preds[indices] == class_idx)
            class_acc = 100 * class_correct / len(indices)
            print(f"  Class '{class_name}': Acc={class_acc:.2f}% ({class_correct}/{len(indices)})")
        else:
            print(f"  Class '{class_name}': No samples present in the test set.")

def save_plots(config_name, history, plot_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'r-', label='Train Loss')
    plt.title(f'{config_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'g-', label='Train Acc')
    plt.plot(history['test_acc'], 'b-', label='Test Acc')
    plt.title(f'{config_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    filename = f"{config_name.lower().replace(' ', '_')}_history.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
    print(f"Saved {config_name} curves to {filename}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Establish organized directories
    src_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(src_dir)
    models_dir = os.path.join(workspace_dir, "models")
    plots_dir = os.path.join(workspace_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load raw dataset
    X_by_label = dataset.load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = dataset.split_80_20(X_by_label)
    
    all_results = {}
    
    # =========================================================================
    # Config 1: Baseline MLP (Amplitude Only)
    # =========================================================================
    print("\n--- Preprocessing: Amplitude Only (Config 1 & 2) ---")
    x_train_amp = dataset.preprocess_csi_amp_only(x_train_raw)
    x_test_amp = dataset.preprocess_csi_amp_only(x_test_raw)
    
    x_train_amp_t = torch.tensor(x_train_amp, dtype=torch.float32).permute(0, 2, 1) # (N, 114, 50)
    x_test_amp_t = torch.tensor(x_test_amp, dtype=torch.float32).permute(0, 2, 1)
    
    train_loader_amp = DataLoader(TensorDataset(x_train_amp_t, torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader_amp = DataLoader(TensorDataset(x_test_amp_t, torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    model_mlp = models.SimpleMLP(input_dim=5700, num_classes=4).to(device)
    best_acc_mlp, weights_mlp, history_mlp = train_configuration(
        "Baseline MLP (Amp Only)", model_mlp, train_loader_amp, test_loader_amp, epochs=20, device=device
    )
    all_results["Baseline MLP (Amp Only)"] = (best_acc_mlp, history_mlp)
    torch.save(weights_mlp, os.path.join(models_dir, "best_baseline_mlp.pth"))
    save_plots("Baseline MLP (Amp Only)", history_mlp, plots_dir)
    evaluate_best_model("Baseline MLP (Amp Only)", model_mlp, weights_mlp, test_loader_amp, device)
    
    # =========================================================================
    # Config 2: Intermediate CNN1D (Amplitude Only)
    # =========================================================================
    model_cnn1d_amp = models.Advanced1DCNN(n_subcarriers=114, num_classes=4).to(device)
    best_acc_cnn_amp, weights_cnn_amp, history_cnn_amp = train_configuration(
        "Intermediate CNN1D (Amp Only)", model_cnn1d_amp, train_loader_amp, test_loader_amp, epochs=20, device=device
    )
    all_results["Intermediate CNN1D (Amp Only)"] = (best_acc_cnn_amp, history_cnn_amp)
    torch.save(weights_cnn_amp, os.path.join(models_dir, "best_intermediate_cnn.pth"))
    save_plots("Intermediate CNN1D (Amp Only)", history_cnn_amp, plots_dir)
    evaluate_best_model("Intermediate CNN1D (Amp Only)", model_cnn1d_amp, weights_cnn_amp, test_loader_amp, device)
    
    # =========================================================================
    # Config 3: Optimized CNN1D (Amplitude + Calibrated Phase Fusion)
    # =========================================================================
    print("\n--- Preprocessing: Amplitude + Calibrated Phase Fusion (Config 3) ---")
    x_train_fusion = dataset.preprocess_csi_fusion(x_train_raw)
    x_test_fusion = dataset.preprocess_csi_fusion(x_test_raw)
    
    x_train_fusion_t = torch.tensor(x_train_fusion, dtype=torch.float32).permute(0, 2, 1) # (N, 228, 50)
    x_test_fusion_t = torch.tensor(x_test_fusion, dtype=torch.float32).permute(0, 2, 1)
    
    train_loader_fusion = DataLoader(TensorDataset(x_train_fusion_t, torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader_fusion = DataLoader(TensorDataset(x_test_fusion_t, torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    model_cnn1d_fusion = models.Advanced1DCNN(n_subcarriers=228, num_classes=4).to(device)
    best_acc_fusion, weights_fusion, history_fusion = train_configuration(
        "Optimized CNN1D (Amp + Phase)", model_cnn1d_fusion, train_loader_fusion, test_loader_fusion, epochs=20, device=device
    )
    all_results["Optimized CNN1D (Amp + Phase)"] = (best_acc_fusion, history_fusion)
    torch.save(weights_fusion, os.path.join(models_dir, "best_optimized_cnn.pth"))
    save_plots("Optimized CNN1D (Amp + Phase)", history_fusion, plots_dir)
    evaluate_best_model("Optimized CNN1D (Amp + Phase)", model_cnn1d_fusion, weights_fusion, test_loader_fusion, device)
    
    # 4. Generate Joint Comparison Plot
    plt.figure(figsize=(10, 6))
    for name, (best_acc, history) in all_results.items():
        plt.plot(range(1, len(history['test_acc']) + 1), history['test_acc'], label=f"{name} (Best: {best_acc:.2f}%)")

    plt.title('Test Accuracy Comparison across Milestones (Strict 80/20 split)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    comparison_filename = "model_comparison.png"
    plt.savefig(os.path.join(plots_dir, comparison_filename))
    plt.close()
    print(f"\nSaved overall comparison plot to {comparison_filename} inside plots/ directory.")
    
    # 5. Print final summary table
    print("\n================ Final Performance Summary ================")
    print(f"{'Configuration':30s} | {'Best Test Accuracy (%)':22s}")
    print("-" * 55)
    for name, (acc, _) in all_results.items():
        print(f"{name:30s} | {acc:22.2f}%")

if __name__ == '__main__':
    main()
