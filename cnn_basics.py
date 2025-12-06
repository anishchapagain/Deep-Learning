# cnn_basics.py
"""
A beginner-friendly example of training a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch.

The script demonstrates the typical workflow:
Import libraries & set device
Load & preprocess the MNIST dataset
Define a simple CNN architecture
Specify loss function and optimizer
Implement the training loop
Evaluate the model on the test set

Run the script with:
```bash
python cnn_basics.py
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# Data loading & preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),               # Convert PIL image to Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Mean & std of MNIST
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer: 1 input channel (gray), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully‑connected layer (after flattening)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST
        self.pool = nn.MaxPool2d(2, 2)  # 2×2 max‑pool

    def forward(self, x):
        # First conv → ReLU → pool
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv → ReLU → pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten for fully‑connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train():
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] – Loss: {avg_loss:.4f}")

# Evaluation
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()
    test()
