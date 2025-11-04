# PyTorch Regression Example: Training, Saving, and Reusing a Model

## 1. Objective
This example demonstrates how to:
1. Build and train a simple regression model in PyTorch.
2. Save the trained model to a `.pth` file.
3. Load and reuse the saved model for predictions or other use cases.

We will create a simple synthetic regression dataset and use a basic feedforward neural network to learn the mapping.

---

## 2. Technical Specifications
| Component | Specification |
|------------|----------------|
| **Framework** | PyTorch 2.x |
| **Task Type** | Regression |
| **Dataset** | Synthetic (y = 2x + 3 + noise) |
| **Model Architecture** | 2-layer Fully Connected Neural Network |
| **Loss Function** | Mean Squared Error (MSELoss) |
| **Optimizer** | Adam |
| **Device** | CPU or GPU (if available) |

---

## 3. Step-by-Step Example

### Step 1: Import Required Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

### Step 2: Generate Synthetic Data
```python
# Create simple linear data: y = 2x + 3 + noise
x = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * x + 3 + torch.randn(x.size()) * 2

# Visualize
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.legend()
plt.show()
```

### Step 3: Define a Simple Regression Model
```python
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RegressionModel()
print(model)
```

### Step 4: Define Loss and Optimizer
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### Step 5: Train the Model
```python
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Step 6: Save the Trained Model
```python
# Save model weights
model_path = 'linear_regression_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
```

---

## 4. Reusing the Saved Model
Now, we simulate loading the model in another use case — for example, making predictions or continuing training.

### Step 1: Load Model Architecture and Weights
```python
# Define same model architecture
loaded_model = RegressionModel()

# Load weights
loaded_model.load_state_dict(torch.load('linear_regression_model.pth', map_location='cpu'))

# Switch to evaluation mode
loaded_model.eval()
```

### Step 2: Use Model for Prediction
```python
# Test data
x_test = torch.tensor([[5.0], [10.0], [-3.0]])
y_pred = loaded_model(x_test)
print("Predictions:")
for inp, out in zip(x_test, y_pred):
    print(f"x = {inp.item():.2f}, predicted y = {out.item():.2f}")
```

### Step 3: Visualize the Fitted Line
```python
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), loaded_model(x).detach().numpy(), color='red', label='Fitted Line')
plt.legend()
plt.show()
```

---

## 5. When and Why to Save & Load Models
| Use Case | Description |
|-----------|-------------|
| **Checkpointing** | Save model progress during training to resume later. |
| **Deployment** | Save a trained model and load it in production for inference. |
| **Transfer Learning** | Reuse trained weights for similar tasks. |
| **Experimentation** | Load different models for performance comparison. |

---

## 6. Summary
- A PyTorch model can be saved using `torch.save(model.state_dict(), path)`.
- To reuse, define the same architecture and load weights using `load_state_dict()`.
- This workflow is common in regression, classification, and deep learning applications.

---

**Next Step:** Try extending this model to multivariate regression or polynomial fitting to explore how PyTorch handles nonlinear mappings.

