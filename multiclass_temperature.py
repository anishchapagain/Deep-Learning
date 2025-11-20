"""
Dataset Description: multi_class.csv

This dataset contains synthetic sensor readings designed for a multi-class classification task.
The goal is to classify the operational state of a machine into three categories.

Features:
- temperature: Sensor reading for temperature.
- vibration: Sensor reading for vibration.
- pressure: Sensor reading for pressure.
- rotation_speed: RPM of the machine.

Target:
- state: Operational state of the machine (0: Normal, 1: Warning, 2: Failure).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Load and Prepare the Dataset ---
print("Step 1: Loading and preparing data...")
# Load data from the CSV file
df = pd.read_csv('multi_class.csv')

# Separate features (X) and the target variable (y)
X = df.drop('state', axis=1).values.astype(np.float32)
y = df['state'].values.astype(np.int64)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Classes found: {np.unique(y)}")

# --- 2. Split Data and Scale Features ---
print("\nStep 2: Splitting data and scaling features...")
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features. It's important to fit the scaler ONLY on the training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")

# --- 3. Convert to PyTorch Tensors and Create DataLoaders ---
print("\nStep 3: Creating PyTorch DataLoaders...")
# Convert numpy arrays to PyTorch tensors
X_train_t = torch.from_numpy(X_train_scaled)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test_scaled)
y_test_t = torch.from_numpy(y_test)

# Create TensorDatasets
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

# Create DataLoaders for batching
BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"DataLoaders created with batch size {BATCH_SIZE}.")

# --- 4. Define the Neural Network for Multi-Class Classification ---
print("\nStep 4: Defining the neural network...")
class MultiClassNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)  # Output layer has a neuron for each class
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # No activation function here; CrossEntropyLoss expects raw logits
        x = self.fc3(x)
        return x

# Instantiate the model
INPUT_DIM = X_train_t.shape[1]
NUM_CLASSES = len(np.unique(y))
model = MultiClassNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
print("Model architecture:")
print(model)

# --- 5. Define Loss Function and Optimizer ---
print("\nStep 5: Defining loss function and optimizer...")
# CrossEntropyLoss is standard for multi-class classification.
# It combines LogSoftmax and NLLLoss, so we don't need a final activation on the model.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Using CrossEntropyLoss and Adam optimizer.")

# --- 6. Training the Model ---
print("\nStep 6: Starting model training...")
NUM_EPOCHS = 50
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        y_pred_logits = model(batch_X)
        
        # Compute loss
        loss = criterion(y_pred_logits, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

print("Training finished.")

# --- 7. Evaluate the Model ---
print("\nStep 7: Evaluating model performance on the test set...")
model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for efficiency
    for batch_X, batch_y in test_loader:
        y_pred_logits = model(batch_X)
        
        # Get the predicted class by finding the index of the max logit
        y_pred_labels = torch.argmax(y_pred_logits, dim=1)
        
        all_preds.extend(y_pred_labels.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Calculate and print metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['State 0: Normal', 'State 1: Warning', 'State 2: Failure']))

# --- 8. Save the Model and Scaler ---
print("\nStep 8: Saving the trained model and scaler...")
torch.save(model.state_dict(), "multiclass_temp_model.pth")
joblib.dump(scaler, "multiclass_temp_scaler.pkl")
print("Model saved to 'multiclass_temp_model.pth'")
print("Scaler saved to 'multiclass_temp_scaler.pkl'")

# --- 9. Perform Inference on New Data ---
print("\nStep 9: Performing inference on new sample data...")

# 1. Load the saved model and scaler
loaded_model = MultiClassNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
loaded_model.load_state_dict(torch.load("multiclass_temp_model.pth"))
loaded_model.eval()

loaded_scaler = joblib.load("multiclass_temp_scaler.pkl")
print("Model and scaler loaded successfully.")

# 2. Define new, unseen data points
# Sample 1: Looks like a 'Normal' state
new_data_normal = np.array([[25.0, 0.2, 105.0, 1000.0]], dtype=np.float32)
# Sample 2: Looks like a 'Warning' state
new_data_warning = np.array([[48.0, 1.0, 140.0, 950.0]], dtype=np.float32)
# Sample 3: Looks like a 'Failure' state
new_data_failure = np.array([[80.0, 4.0, 190.0, 1150.0]], dtype=np.float32)

# 3. Preprocess the new data using the loaded scaler
new_data_normal_scaled = loaded_scaler.transform(new_data_normal)
new_data_warning_scaled = loaded_scaler.transform(new_data_warning)
new_data_failure_scaled = loaded_scaler.transform(new_data_failure)

# 4. Convert to PyTorch tensors
new_tensor_normal = torch.from_numpy(new_data_normal_scaled)
new_tensor_warning = torch.from_numpy(new_data_warning_scaled)
new_tensor_failure = torch.from_numpy(new_data_failure_scaled)

# 5. Make predictions
with torch.no_grad():
    pred_normal_logits = loaded_model(new_tensor_normal)
    pred_warning_logits = loaded_model(new_tensor_warning)
    pred_failure_logits = loaded_model(new_tensor_failure)

    pred_normal_label = torch.argmax(pred_normal_logits, dim=1).item()
    pred_warning_label = torch.argmax(pred_warning_logits, dim=1).item()
    pred_failure_label = torch.argmax(pred_failure_logits, dim=1).item()

print(f"\nPrediction for Normal sample {new_data_normal[0]}: State {pred_normal_label}")
print(f"Prediction for Warning sample {new_data_warning[0]}: State {pred_warning_label}")
print(f"Prediction for Failure sample {new_data_failure[0]}: State {pred_failure_label}")