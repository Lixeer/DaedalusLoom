import torch
import torch.nn as nn

class Advanced1DCNN(nn.Module):
    def __init__(self, n_subcarriers=228, num_classes=4):
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
        # input shape: (batch, channels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5700, num_classes=4): # 50 * 114 = 5700, 50 * 228 = 11400
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
        # input shape: (batch, channels, time)
        return self.net(x)

class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_dim=228, hidden_dim=128, num_layers=2, num_classes=4):
        super(LSTMGestureClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # input shape: (batch, channels, time)
        # nn.LSTM expects: (batch, time, channels)
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Concatenate bidirectional hidden states of the last layer
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(last_hidden)
