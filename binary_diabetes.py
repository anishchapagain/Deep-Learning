import openml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Load the dataset from OpenML
dataset = openml.datasets.get_dataset(37)  # Pima‐indians diabetes dataset :contentReference[oaicite:1]{index=1}
X, y, categorical_indicator, feature_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

# Explore the dataset (optional)
print("Feature names:", feature_names)
print("First 5 rows of X:\n", X.head())
print("First 5 labels of y:\n", y.head())
print("Categorical indicators:", categorical_indicator)
print(pd.DataFrame(X, columns=feature_names).describe())
print(pd.DataFrame(X, columns=feature_names).head())
quit(1)  # Remove this line after exploration

# Convert to numpy / proper types
X = X.to_numpy().astype(np.float32)
# y might be strings like '0'/'1' or int — convert appropriately:
y = y.map({"tested_negative": 0, "tested_positive": 1}).to_numpy().astype(np.int64)
# y = y.to_numpy().astype(np.int64)  # PyTorch expects integer labels for classification

# 2. Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Convert to torch tensors
X_train_t = torch.from_numpy(X_train_scaled)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test_scaled)
y_test_t = torch.from_numpy(y_test)

# 5. Create Dataset and DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 6. Define a simple neural network for binary classification
class DiabetesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)  # output: single logit
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out_act(self.fc3(x))
        return x

input_dim = X_train_t.shape[1]
model = DiabetesNet(input_dim)

# 7. Loss, optimizer
criterion = nn.BCELoss()  # binary cross-entropy, since using Sigmoid
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X).squeeze(1)  # shape: (batch,)
        loss = criterion(y_pred, batch_y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 9. Save model + scaler
torch.save(model.state_dict(), "diabetes_model.pth")
import joblib
joblib.dump(scaler, "scaler.pkl")

# 10. Inference / Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X).squeeze(1)
        all_preds.append(y_pred.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Convert probabilities → binary labels (threshold = 0.5)
y_pred_labels = (all_preds >= 0.5).astype(int)

# Simple accuracy:
accuracy = (y_pred_labels == all_labels).mean()
print("Test Accuracy:", accuracy)

# 11. Example: Inference on new data
# Suppose you have a new numpy array `X_new` of shape (n_samples, n_features)
X_new = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],  # example row
                  [1, 85, 66, 29, 0, 26.6, 0.351, 31]], dtype=np.float32)
# Scale it:
scaler = joblib.load("scaler.pkl")
X_new_scaled = scaler.transform(X_new)
X_new_t = torch.from_numpy(X_new_scaled)

model = DiabetesNet(input_dim)
model.load_state_dict(torch.load("diabetes_model.pth"))
model.eval()

with torch.no_grad():
    y_new_pred = model(X_new_t).squeeze(1).cpu().numpy()

print("Predicted probabilities for new data:", y_new_pred)
print("Predicted labels:", (y_new_pred >= 0.5).astype(int))