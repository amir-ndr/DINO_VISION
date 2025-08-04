import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Layer a: Convolutional layer (32 filters, 5x5 kernel, stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Layer c: Max pooling (2x2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer d: Convolutional layer (64 filters, 5x5 kernel, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # Layer g: Fully connected layer (128 neurons)
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=128)
        # Layer i: Dropout (p=0.5)
        self.dropout = nn.Dropout(p=0.5)
        # Layer j: Fully connected layer (10 neurons)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)               # Layer a
        x = F.relu(x)                   # Layer b
        x = self.pool(x)                # Layer c
        x = self.conv2(x)               # Layer d
        x = F.relu(x)                   # Layer e
        x = self.pool(x)                # Layer f
        x = x.view(x.size(0), -1)       # Flatten for fully connected layer
        x = F.relu(self.fc1(x))         # Layer h
        x = self.dropout(x)             # Layer i
        x = self.fc2(x)                 # Layer j
        x = F.softmax(x, dim=1)         # Layer k
        return x

model = CNNModel()

# Sample input of size (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 10, 10)  # (batch_size=1, channels=3, height=10, width=10)

# Forward pass
output = model(input_tensor)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Print the number of parameters
total_params = count_parameters(model)
print(f'Total number of parameters: {total_params}')