
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Step 1: Load and Prepare the Dataset
# We use pandas to read the CSV file into a DataFrame.
data = pd.read_csv('data.csv')

# Display the first few rows and plot the raw data
print("Data sample:")
print(data.head())

plt.figure(figsize=(10, 5))
plt.scatter(data['Celsius'], data['Fahrenheit'], label='Original data')
plt.title('Celsius vs Fahrenheit (Raw Data)')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.legend()
plt.show()

# Extract Celsius and Fahrenheit columns for processing
celsius = data[['Celsius']].values
fahrenheit = data[['Fahrenheit']].values

# --- Feature Scaling ---
# It's crucial to scale input features for neural networks. We'll use StandardScaler
# to transform our data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
celsius_scaled = scaler.fit_transform(celsius)

# Save the scaler to a file. This is important because we need to use the exact
# same scaling transformation on any new data before making predictions.
scaler_path = 'celsius_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(celsius_scaled, dtype=torch.float32)
y_train = torch.tensor(fahrenheit, dtype=torch.float32)

# Create a TensorDataset and DataLoader
# Increasing batch_size provides more stable gradient estimates during training.
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 2: Define the Neural Network Model (No changes needed here)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()

# Step 3: Define Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Learning rate can be slightly higher with scaled data

# Step 4: Train the Model
# Increased epochs for better convergence on noisy data.
num_epochs = 500
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Model Performance Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_train)
    train_loss = loss_fn(predictions, y_train)
    print(f'\nMean Squared Error on Training Data: {train_loss.item():.4f}')

# You can also see the learned parameters. These are based on the SCALED data.
[w, b] = model.parameters()
print(f'Learned weight (on scaled data): {w.item():.4f}, Learned bias: {b.item():.4f}')

# To draw a clean regression line, we need to plot against a sorted range of X values.
# 1. Get the original (unscaled) Celsius values for plotting.
x_plot = X_train.numpy()
# 2. Get the model's predictions.
y_plot = predictions.numpy()

# 3. Create a combined array and sort by the x-values (Celsius).
# This prevents matplotlib from drawing a jagged line.
plot_data = sorted(zip(x_plot, y_plot))
x_sorted, y_sorted = zip(*plot_data)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.scatter(celsius, fahrenheit, label='Original Data Points', alpha=0.7)
plt.plot(scaler.inverse_transform(np.array(x_sorted).reshape(-1, 1)), y_sorted, label='Fitted Regression Line', color='r', linewidth=2)
plt.title('Model Performance: Original Data vs. Fitted Line')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Save the Model
model_path = 'linear_regression_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


print("\n--- Standalone Model Usage Example ---")
print("This shows how to load the model AND the scaler to make predictions on new raw data.")

# 1. Define the Model Architecture (must be identical)
class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 2. Load the Scaler
inference_scaler = joblib.load(scaler_path)

# 3. Instantiate the model and load the trained weights
inference_model = InferenceModel()
inference_model.load_state_dict(torch.load(model_path))
inference_model.eval()

# 4. Prepare new raw data
new_celsius_values = np.array([[-15], [0], [10], [25], [35]], dtype=np.float32)

# 5. VERY IMPORTANT: Scale the new data using the loaded scaler
scaled_celsius = inference_scaler.transform(new_celsius_values)

# 6. Convert to tensor and make predictions
celsius_tensor = torch.tensor(scaled_celsius, dtype=torch.float32)
with torch.no_grad():
    predictions_fahrenheit = inference_model(celsius_tensor)

# 7. Display results
print("\nPredictions on new raw Celsius values:")
for i, c_val in enumerate(new_celsius_values):
    print(f"Celsius: {c_val.item():.1f}, Predicted Fahrenheit: {predictions_fahrenheit[i].item():.2f}")

