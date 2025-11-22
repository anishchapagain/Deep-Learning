# Neural Network Regression with PyTorch

## This notebook demonstrates **how a simple neural network learns a regression task** using **PyTorch**.

### We'll cover:
<!-- 1. Creating synthetic data for regression  
2. Defining a neural network model  
3. Training the model using a loss function and optimizer  
4. Visualizing the learning process   -->

## Step 1: Import Libraries

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## Step 2: Generate Synthetic Data
# Set random seed for reproducibility
torch.manual_seed(42) 

# Generate synthetic data 
X = torch.linspace(-5, 5, 100).unsqueeze(1) # shape [100, 1] 
y = 3 * X + 2 + torch.randn(X.size()) * 1.5 # add noise 

# Visualize 
plt.scatter(X.numpy(), y.numpy(), label='Data') 
plt.title("Generated Synthetic Data") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.legend() 
plt.show()
## Step 3: Define a Simple Neural Network We'll use a **feedforward neural network** with: - Input layer: 1 neuron (for x) - Hidden layer: 10 neurons (with ReLU) - Output layer: 1 neuron (for predicted y)
class RegressionNN(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.net = nn.Sequential( nn.Linear(1, 10), 
        nn.ReLU(), 
        nn.Linear(10, 1) 
        ) 
    def forward(self, x): 
        return self.net(x) 

# Instantiate model 
model = RegressionNN() 
print(model)

## Step 4: Define Loss Function and Optimizer We'll use: 
# - **Mean Squared Error (MSE)** for regression loss 
# - **Adam optimizer** for efficient learning 
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

## Step 5: Training the Model The training loop involves: 
# 1. Forward pass — compute predictions 
# 2. Compute loss (MSE between predictions and actual y) 
# 3. Backward pass — compute gradients 
# 4. Update parameters
epochs = 500 
losses = [] 
for epoch in range(epochs): 
    # Forward pass 
    y_pred = model(X) 
    loss = criterion(y_pred, y) 
    # Backward pass and optimization 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    losses.append(loss.item()) 

    # Print progress 
    if (epoch+1) % 100 == 0: 
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

## Step 6: Visualize Loss Curve
plt.plot(losses) 
plt.title("Training Loss Curve") 
plt.xlabel("Epoch") 
plt.ylabel("MSE Loss") 
plt.show()

## Step 7: Compare Predictions vs True Data
# Evaluate model 
model.eval() 
with torch.no_grad(): 
    predicted = model(X) 
plt.scatter(X.numpy(), y.numpy(), label='True Data') 
plt.plot(X.numpy(), predicted.numpy(), color='red', label='Model Prediction') 
plt.title("Model Prediction vs True Data") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.legend() 
plt.show()

## Step 8: Summary We built and trained a **simple neural network for regression** using PyTorch. Key takeaways: - Neural networks can approximate even simple functions like y = 3x + 2. - The loss function guides the model to minimize prediction error. - With each epoch, parameters update to better fit the data. 

# Try experimenting by: - Adding more layers or neurons. - Changing activation functions (e.g., `Tanh`). - Modifying the learning rate or noise level.