import torch
import torch.nn as nn

class FlagCNN(nn.Module):
    """
    1D CNN for flag pattern classification.
    Input: (batch, window, features) = (batch, 64, 5)
    Output: (batch, num_classes) = (batch, 7)
    """
    def __init__(self, num_features=5, num_classes=7):
        super(FlagCNN, self).__init__()
        
        # Conv layers expect (batch, channels, sequence_length)
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x: (batch, window, features) -> transpose to (batch, features, window)
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
