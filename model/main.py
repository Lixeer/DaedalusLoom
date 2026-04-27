import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# 1. 数据加载与预处理
def load_and_preprocess():
    try:
        amp1 = np.load("filter_dataset/amp/fist_data.npz", allow_pickle=True)['data']
        pha1 = np.load("filter_dataset/pha/fist_data.npz", allow_pickle=True)['data']
        amp2 = np.load("filter_dataset/amp/paper_data.npz", allow_pickle=True)['data']
        pha2 = np.load("filter_dataset/pha/paper_data.npz", allow_pickle=True)['data']
        amp3 = np.load("filter_dataset/amp/scissor_data.npz", allow_pickle=True)['data']
        pha3 = np.load("filter_dataset/pha/scissor_data.npz", allow_pickle=True)['data']
        print("成功加载本地 .npz 数据集")
    except FileNotFoundError:
        print("未找到文件")

    def format_to_torch(data):
        data = data.reshape(-1, 100, 2, 166)
        data = np.transpose(data, (0, 2, 1, 3))
        return torch.tensor(data)

    X = torch.cat([format_to_torch(np.concatenate((amp1, pha1), axis=2).astype(np.float32)), 
                   format_to_torch(np.concatenate((amp2, pha2), axis=2).astype(np.float32)), 
                   format_to_torch(np.concatenate((amp3, pha3), axis=2).astype(np.float32))], dim=0)
    Y = torch.cat([torch.zeros(amp1.shape[0], dtype=torch.long), 
                   torch.ones(amp2.shape[0], dtype=torch.long), 
                   torch.full((amp3.shape[0],), 2, dtype=torch.long)])

    return train_test_split(X, Y, test_size=0.2, random_state=42)

#多层感知机
class SimpleMLP(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.input_dim = 2 * 100 * 166
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)
class CSIGestureCNN(nn.Module):

    def __init__(self, num_classes=3):

        super(CSIGestureCNN, self).__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(2, 32, kernel_size=3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 25 * 41, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)

        )



    def forward(self, x):

        x = self.conv_block(x)

        x = self.classifier(x)

        return x



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test = load_and_preprocess()
    
    print(f"训练集大小: {x_train.shape[0]}")  # 80%
    print(f"测试集大小: {x_test.shape[0]}")    # 20%
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=200, shuffle=False)
    model = SimpleMLP(num_classes=3).to(device)
    #model = CSIGestureCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    epochs = 100
    print(f"\n开始在 {device} 上训练...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 修改 2：计算训练集准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        train_accuracy = 100 * train_correct / train_total
        
        # 计算测试准确率
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        avg_loss = running_loss / len(train_loader)
        
        # 保存历史记录
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        
        print(f"-> Epoch Summary: Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")


    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], 'r-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 绘制 Accuracy 曲线 (同时显示训练和测试)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], 'g-', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['test_acc'], 'b-', label='Test Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()