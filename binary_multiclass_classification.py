# PyTorch Classification Examples: Binary and Multi-Class
# This notebook demonstrates classification tasks using PyTorch with real datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ============================================================================
# PART 1: BINARY CLASSIFICATION (Make Moons Dataset)
# ============================================================================

print("\n" + "="*70)
print("PART 1: BINARY CLASSIFICATION")
print("="*70)

# Generate binary classification dataset (two interleaving half circles)
X_binary, y_binary = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Split the data
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Standardize features
scaler_binary = StandardScaler()
X_train_b = scaler_binary.fit_transform(X_train_b)
X_test_b = scaler_binary.transform(X_test_b)

# Convert to PyTorch tensors
X_train_b_tensor = torch.FloatTensor(X_train_b)
y_train_b_tensor = torch.FloatTensor(y_train_b).unsqueeze(1)
X_test_b_tensor = torch.FloatTensor(X_test_b)
y_test_b_tensor = torch.FloatTensor(y_test_b).unsqueeze(1)

# Create DataLoaders
train_dataset_b = TensorDataset(X_train_b_tensor, y_train_b_tensor)
test_dataset_b = TensorDataset(X_test_b_tensor, y_test_b_tensor)
train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True)
test_loader_b = DataLoader(test_dataset_b, batch_size=32, shuffle=False)

# Visualize binary classification dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
scatter = ax.scatter(X_train_b[:, 0], X_train_b[:, 1], c=y_train_b, 
                     cmap='viridis', alpha=0.6, edgecolors='k', s=50)
ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('Binary Classification Dataset (Make Moons)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Class')
plt.tight_layout()
plt.show()

# Define binary classification neural network
class BinaryClassifier(nn.Module):
    """
    Neural network for binary classification.
    Architecture: Input(2) -> Hidden(16) -> Hidden(8) -> Output(1)
    Uses ReLU activation and Sigmoid output for binary classification.
    """
    def __init__(self, input_size=2, hidden_size1=16, hidden_size2=8):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss function, and optimizer
binary_model = BinaryClassifier()
criterion_binary = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer_binary = optim.Adam(binary_model.parameters(), lr=0.01)

print(f"\nBinary Model Architecture:\n{binary_model}")

# Training function for binary classification
def train_binary_model(model, train_loader, criterion, optimizer, epochs=100):
    """Train the binary classification model and track loss history."""
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return loss_history

# Train binary model
print("\nTraining Binary Classification Model...")
binary_loss_history = train_binary_model(
    binary_model, train_loader_b, criterion_binary, optimizer_binary, epochs=100
)

# Evaluate binary model
def evaluate_binary_model(model, test_loader):
    """Evaluate the binary classification model on test data."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.numpy())
            all_labels.extend(batch_y.numpy())
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_labels

accuracy_binary, preds_binary, labels_binary = evaluate_binary_model(
    binary_model, test_loader_b
)
print(f"\nBinary Classification Test Accuracy: {accuracy_binary:.2f}%")

# Visualize binary classification results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Training loss
axes[0].plot(binary_loss_history, color='blue', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Binary Classification Training Loss', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Decision boundary
h = 0.02  # Step size in the mesh
x_min, x_max = X_train_b[:, 0].min() - 0.5, X_train_b[:, 0].max() + 0.5
y_min, y_max = X_train_b[:, 1].min() - 0.5, X_train_b[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = binary_model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.detach().numpy().reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
axes[1].scatter(X_test_b[:, 0], X_test_b[:, 1], c=y_test_b, 
                cmap='RdYlBu', edgecolors='k', s=50)
axes[1].set_xlabel('Feature 1', fontsize=12)
axes[1].set_ylabel('Feature 2', fontsize=12)
axes[1].set_title('Decision Boundary (Binary Classification)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# PART 2: MULTI-CLASS CLASSIFICATION (Wine Dataset)
# ============================================================================

print("\n" + "="*70)
print("PART 2: MULTI-CLASS CLASSIFICATION")
print("="*70)

# Load Wine dataset (3 classes, 13 features)
wine = load_wine()
X_multi = wine.data
y_multi = wine.target

print(f"\nWine Dataset Info:")
print(f"Number of samples: {X_multi.shape[0]}")
print(f"Number of features: {X_multi.shape[1]}")
print(f"Number of classes: {len(np.unique(y_multi))}")
print(f"Class distribution: {np.bincount(y_multi)}")

# Split the data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Standardize features
scaler_multi = StandardScaler()
X_train_m = scaler_multi.fit_transform(X_train_m)
X_test_m = scaler_multi.transform(X_test_m)

# Convert to PyTorch tensors
X_train_m_tensor = torch.FloatTensor(X_train_m)
y_train_m_tensor = torch.LongTensor(y_train_m)  # LongTensor for class indices
X_test_m_tensor = torch.FloatTensor(X_test_m)
y_test_m_tensor = torch.LongTensor(y_test_m)

# Create DataLoaders
train_dataset_m = TensorDataset(X_train_m_tensor, y_train_m_tensor)
test_dataset_m = TensorDataset(X_test_m_tensor, y_test_m_tensor)
train_loader_m = DataLoader(train_dataset_m, batch_size=16, shuffle=True)
test_loader_m = DataLoader(test_dataset_m, batch_size=16, shuffle=False)

# Define multi-class classification neural network
class MultiClassClassifier(nn.Module):
    """
    Neural network for multi-class classification.
    Architecture: Input(13) -> Hidden(64) -> Hidden(32) -> Output(3)
    Uses ReLU activation and outputs raw logits (no activation on final layer).
    """
    def __init__(self, input_size=13, hidden_size1=64, hidden_size2=32, num_classes=3):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Regularization
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation - CrossEntropyLoss expects raw logits
        return x

# Initialize model, loss function, and optimizer
multi_model = MultiClassClassifier()
criterion_multi = nn.CrossEntropyLoss()  # For multi-class classification
optimizer_multi = optim.Adam(multi_model.parameters(), lr=0.001)

print(f"\nMulti-Class Model Architecture:\n{multi_model}")

# Training function for multi-class classification
def train_multi_model(model, train_loader, criterion, optimizer, epochs=200):
    """Train the multi-class classification model and track loss and accuracy."""
    model.train()
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return loss_history, accuracy_history

# Train multi-class model
print("\nTraining Multi-Class Classification Model...")
multi_loss_history, multi_acc_history = train_multi_model(
    multi_model, train_loader_m, criterion_multi, optimizer_multi, epochs=200
)

# Evaluate multi-class model
def evaluate_multi_model(model, test_loader):
    """Evaluate the multi-class classification model on test data."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_labels

accuracy_multi, preds_multi, labels_multi = evaluate_multi_model(
    multi_model, test_loader_m
)
print(f"\nMulti-Class Classification Test Accuracy: {accuracy_multi:.2f}%")

# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(labels_multi, preds_multi)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(labels_multi, preds_multi, target_names=wine.target_names))

# Visualize multi-class classification results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training loss
axes[0, 0].plot(multi_loss_history, color='red', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Multi-Class Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training accuracy
axes[0, 1].plot(multi_acc_history, color='green', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 1].set_title('Multi-Class Training Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=wine.target_names, yticklabels=wine.target_names)
axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
axes[1, 0].set_ylabel('True Label', fontsize=12)
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 4: Class-wise accuracy
class_accuracies = []
for i in range(len(wine.target_names)):
    class_mask = np.array(labels_multi) == i
    class_correct = np.sum((np.array(preds_multi)[class_mask] == np.array(labels_multi)[class_mask]))
    class_total = np.sum(class_mask)
    class_acc = 100 * class_correct / class_total if class_total > 0 else 0
    class_accuracies.append(class_acc)

axes[1, 1].bar(wine.target_names, class_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1, 1].set_xlabel('Wine Class', fontsize=12)
axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1, 1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_ylim([0, 105])
for i, v in enumerate(class_accuracies):
    axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Binary Classification Accuracy: {accuracy_binary:.2f}%")
print(f"Multi-Class Classification Accuracy: {accuracy_multi:.2f}%")
print("\nKey Differences:")
print("- Binary: BCELoss with Sigmoid activation, outputs 1 probability")
print("- Multi-Class: CrossEntropyLoss with raw logits, outputs 3 class scores")
print("="*70)