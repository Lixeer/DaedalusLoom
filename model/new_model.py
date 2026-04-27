from numpy import dtype
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def complex_to_real(frame):
    return np.concatenate((
        np.abs(frame).astype(np.float32),  # 振幅
        np.angle(frame).astype(np.float32)), axis=0)  # 相位


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

'''
class CSIModel(nn.Module):
    def __init__(self, input_dim=60, d_model=128, n_head=8, num_layers=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 16, input_dim * 4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, 3),
        )
        self.classifier = nn.Linear(input_dim*4, 3)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        if mask is not None: pass

        return self.classifier(x)
'''
class CSIModel(nn.Module):
    def __init__(self, input_dim=60, d_model=128, n_head=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, 3)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer 编码 (使用 batch_first=True)
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, d_model)

        # 池化（考虑 mask）
        if mask is not None:
            # mask: (batch, seq_len), True 表示需要 mask 的位置
            valid_mask = (~mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
            x = (x * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)  # 避免除零
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


class CSIDataset(Dataset):
    def __init__(self, data1, data2, data3,
                 clean, # 滤波数据去掉这行
                 is_train=True, train_ratio=0.65, seed=42):
        self.keep_indices = np.where(np.array(clean) == False)[0]  # 滤波数据去掉这行

        data_list = []
        labels = []

        for label_idx, data in enumerate([data1, data2, data3]):
            # Shuffle the data before splitting
            if seed is not None:
                np.random.seed(seed + label_idx)  # Different seed for each class
            indices = np.random.permutation(len(data))
            shuffled_data = data[indices]

            # Split after shuffling
            n = len(shuffled_data)
            split_idx = int(n * train_ratio)
            split_data = shuffled_data[:split_idx] if is_train else shuffled_data[split_idx:]
            data_list.append(split_data)
            labels.append(np.full(len(split_data), label_idx, dtype=int))

        self.is_train = is_train
        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][:, self.keep_indices]

        data = np.concatenate([
            np.abs(data).astype(np.float32),
            np.angle(data).astype(np.float32)
        ], axis=1)
        data = (data - data.mean()) / (data.std() + 1e-8)

        return torch.from_numpy(data), self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = lengths.max()

    padded = []
    for seq in sequences:
        # 转换为 tensor
        seq_tensor = torch.from_numpy(seq) if isinstance(seq, np.ndarray) else seq
        # 填充
        if len(seq_tensor) < max_len:
            padding = torch.zeros(max_len - len(seq_tensor), seq_tensor.shape[1])
            seq_tensor = torch.cat([seq_tensor, padding])
        padded.append(seq_tensor)

    padded = torch.stack(padded)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    return padded, torch.tensor(labels), mask


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for data, labels, mask in pbar:
        data, labels, mask = data.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()
        output = model(data, mask)
        loss = criterion(output, labels)
        loss.backward()

        # 计算梯度统计信息
        grad_norm = 0.0
        max_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
                max_grad = max(max_grad, p.grad.abs().max().item())
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += len(labels)

        # 实时更新进度条显示
        current_loss = total_loss / (pbar.n + 1)
        current_acc = correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'grad_norm': f'{grad_norm:.2e}',
            'max_grad': f'{max_grad:.2e}'
        })

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for data, labels, mask in pbar:
            data, labels, mask = data.to(device), labels.to(device), mask.to(device)
            output = model(data, mask)
            loss = criterion(output, labels)

            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()
            total += len(labels)

            # 实时更新进度条显示
            current_loss = total_loss / (pbar.n + 1)
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })

    return total_loss / len(loader), correct / total


def plot_history(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train():
    # 数据加载
    # 滤波数据
    train_set = CSIDataset(data1, data2, data3, clean, is_train=True)
    val_set = CSIDataset(data1, data2, data3, clean, is_train=False)
    # 不滤波数据
    # train_set = CSIDataset(data1, data2, data3, is_train=True)
    # val_set = CSIDataset(data1, data2, data3, is_train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 训练
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0

    for epoch in range(1, epochs + 1):
        # print(f'\nEpoch {epoch}/{epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'✓ Saved best model (acc: {val_acc:.4f})')
        else:
            torch.save(model.state_dict(), f'models/epoch{epoch}.pth')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        plot_history(train_losses, train_accs, val_losses, val_accs)
    print(f'\nBest validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    '''
    # 滤波数据
    amp1 = np.load("dataset4/amp/fist_data.npz", allow_pickle=True)['data']
    pha1 = np.load("dataset4/pha/fist_data.npz", allow_pickle=True)['data']
    amp2 = np.load("dataset4/amp/paper_data.npz", allow_pickle=True)['data']
    pha2 = np.load("dataset4/pha/paper_data.npz", allow_pickle=True)['data']
    amp3 = np.load("dataset4/amp/scissor_data.npz", allow_pickle=True)['data']
    pha3 = np.load("dataset4/pha/scissor_data.npz", allow_pickle=True)['data']
    data1 = np.concatenate((amp1, pha1), axis=2, dtype=np.float32)
    data2 = np.concatenate((amp2, pha2), axis=2, dtype=np.float32)
    data3 = np.concatenate((amp3, pha3), axis=2, dtype=np.float32)
    '''
    # 不滤波数据
    #data1 = np.load("dataset3/fist_complex_data.npz", allow_pickle=True)['data']
    #data2 = np.load("dataset3/paper_complex_data.npz", allow_pickle=True)['data']
    #data3 = np.load("dataset3/scissor_complex_data.npz", allow_pickle=True)['data']
    # 滤波数据去掉这行
    #clean = [True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, ]


    lr = 1e-5
    batch_size = 32
    epochs = 20
    # 模型
    model = CSIModel(input_dim=332, d_model=128, n_head=4, num_layers=2).to(device)
    # 读训练好的模型，想微调或者预测把这行注释去了
    model.load_state_dict(torch.load('models/no_filter_best_model.pth', map_location=device))
    print(device)

    # 如果想训练
    # train()

    # 假如采集到新的数据
    model.eval() # 开启推理模式，模型不会学习
    result = model(  ) # result就是预测向量，想转成概率向量就过一遍softmax，取argmax就是预测结果
