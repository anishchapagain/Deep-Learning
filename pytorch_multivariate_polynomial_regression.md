# Multivariate and Polynomial Regression with PyTorch

## 1. Objective
This guide extends the previous simple regression example to demonstrate how PyTorch can handle **multivariate regression** (multiple input features) and **polynomial regression** (nonlinear relationships between inputs and outputs).

You will learn:
1. How to create synthetic multivariate and polynomial datasets.
2. How to train neural networks for these tasks.
3. How to visualize and interpret model behavior.

---

## 2. Technical Specifications
| Component | Specification |
|------------|----------------|
| **Framework** | PyTorch 2.x |
| **Task Type** | Multivariate & Polynomial Regression |
| **Dataset** | Synthetic (generated using NumPy & PyTorch) |
| **Model Architecture** | Feedforward Neural Network (3–4 layers) |
| **Loss Function** | Mean Squared Error (MSELoss) |
| **Optimizer** | Adam |
| **Device** | CPU or GPU (if available) |

---

## 3. Multivariate Regression Example

### Step 1: Import Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
```

### Step 2: Generate Synthetic Multivariate Data
Let’s define a function:  
\[ y = 3x_1 + 2x_2 - 4x_3 + 5 + \text{noise} \]

```python
# Generate synthetic data
np.random.seed(42)
X = np.random.rand(200, 3)  # 3 input features
y = 3*X[:,0] + 2*X[:,1] - 4*X[:,2] + 5 + np.random.randn(200) * 0.2

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
```

### Step 3: Define the Model
```python
class MultiRegressionModel(nn.Module):
    def __init__(self):
        super(MultiRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MultiRegressionModel()
```

### Step 4: Train the Model
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
```

### Step 5: Evaluate
```python
model.eval()
y_pred = model(X_tensor).detach().numpy()
plt.scatter(y, y_pred, color='blue')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Multivariate Regression Fit')
plt.show()
```

---

## 4. Polynomial Regression Example

### Step 1: Generate Polynomial Data
Let’s model a **nonlinear** function:
\[ y = 0.5x^3 - 2x^2 + 3x + 1 + \text{noise} \]

```python
# Create synthetic polynomial data
x = torch.linspace(-5, 5, 200).reshape(-1, 1)
y = 0.5 * x**3 - 2 * x**2 + 3 * x + 1 + torch.randn_like(x) * 2
```

### Step 2: Define a Deeper Model for Nonlinearity
```python
class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

poly_model = PolynomialRegressionModel()
```

### Step 3: Train the Polynomial Model
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(poly_model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = poly_model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
```

### Step 4: Visualize Results
```python
poly_model.eval()
with torch.no_grad():
    y_fit = poly_model(x)

plt.scatter(x.numpy(), y.numpy(), label='Data', color='gray')
plt.plot(x.numpy(), y_fit.numpy(), color='red', label='Fitted Curve')
plt.legend()
plt.title('Polynomial Regression Fit')
plt.show()
```

---

## 5. Saving and Reusing the Model
```python
# Save models
torch.save(model.state_dict(), 'multivariate_regression.pth')
torch.save(poly_model.state_dict(), 'polynomial_regression.pth')

# Load and reuse
loaded_poly_model = PolynomialRegressionModel()
loaded_poly_model.load_state_dict(torch.load('polynomial_regression.pth'))
loaded_poly_model.eval()
```

---

## 6. Insights and Comparison
| Aspect | Multivariate Regression | Polynomial Regression |
|--------|--------------------------|------------------------|
| Input Type | Multiple independent variables | Single variable with nonlinear dependency |
| Model Depth | Shallow (2–3 layers) | Deeper for nonlinear mapping |
| Typical Use | Predict outcomes based on multiple factors | Fit curved data trends |
| Visualization | 3D or feature-level plots | 2D curve fitting |

---

## 7. Summary
- **Multivariate Regression** handles multiple input features simultaneously.
- **Polynomial Regression** models nonlinear relationships using neural networks.
- PyTorch simplifies both with a unified API for model building, training, and saving.
- Neural networks automatically learn nonlinear transformations without manually adding polynomial terms.

**Next Step:** Try combining both — use multiple nonlinear inputs for a complex regression model and observe how neural networks adapt to higher-dimensional feature spaces.

