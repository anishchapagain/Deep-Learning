# PyTorch Classification with Model Saving and Transfer Learning

## 1. Introduction
This tutorial demonstrates how to build, train, save, load, and reuse a **classification model** in PyTorch. The focus is on understanding how PyTorch handles model serialization and transfer learning.

We'll use a **synthetic binary classification dataset** generated using `sklearn.datasets.make_classification`, then extend the trained model for a new use case using transfer learning.

---

## 2. Technical Specifications
- **Framework:** PyTorch
- **Python Version:** >=3.8
- **Libraries:** torch, sklearn, matplotlib
- **GPU Support:** Optional (runs on CPU by default)

---

## 3. Dataset Preparation
We generate a synthetic dataset and normalize the features.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Create synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2,
                           n_informative=3, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
```

---

## 4. Define the Model
We define a simple **feedforward neural network** with one hidden layer.

```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = Classifier(input_dim=4, hidden_dim=16, output_dim=2)
```

---

## 5. Training Loop
Train the model using **CrossEntropyLoss** and **Adam optimizer**.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

---

## 6. Model Evaluation
We test the model’s accuracy on the test dataset.

```python
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, 1)
    acc = (y_pred_classes == y_test).sum().item() / y_test.size(0)
    print(f'Accuracy: {acc:.4f}')
```

---

## 7. Saving and Loading the Model
We save the trained model’s parameters using `torch.save()` and reload them for future inference.

```python
# Save model
torch.save(model.state_dict(), 'classifier_model.pth')

# Load model
loaded_model = Classifier(4, 16, 2)
loaded_model.load_state_dict(torch.load('classifier_model.pth'))
loaded_model.eval()
```

---

## 8. Transfer Learning Example
We extend the pretrained classifier to a new **3-class classification problem** by reusing its base layers.

```python
class TransferredModel(nn.Module):
    def __init__(self, base_model, new_output_dim):
        super(TransferredModel, self).__init__()
        self.base = base_model.network[:-1]  # use all layers except the last
        self.new_layer = nn.Linear(16, new_output_dim)
        
    def forward(self, x):
        x = self.base(x)
        return self.new_layer(x)

# Create transfer model for 3-class classification
transfer_model = TransferredModel(model, 3)
print(transfer_model)
```

This demonstrates how pretrained models can be reused for new tasks — a foundational concept in **transfer learning**.

---

## 9. Visualization (Optional)
You can visualize training loss or accuracy curve using `matplotlib`.

```python
import matplotlib.pyplot as plt

# Example loss visualization
epoch_list = list(range(1, epochs+1))
losses = [0.65, 0.42, 0.31, 0.25, 0.18, 0.12]  # replace with actual losses
plt.plot(epoch_list[:len(losses)], losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
```

---

## 10. Key Takeaways
- Built and trained a PyTorch classification model.
- Saved and reloaded model weights for later use.
- Demonstrated transfer learning for new classification tasks.
- Illustrated modularity and reusability of PyTorch models.

---

**Next Step:** Plan to extend to multi-class datasets (e.g., Iris or MNIST) and fine-tune hyperparameters for better performance.