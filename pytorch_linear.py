import torch
import torch.nn as nn

# Define a linear layer
linear_layer = nn.Linear(in_features=10, out_features=5)

# Create some input data
input_data = torch.randn(1, 10)

print("Input shape:", input_data.shape)
print("Input data:", input_data)

# Pass the input through the linear layer
output = linear_layer(input_data)

# 'output' now contains the result of the linear transformation,
# effectively acting as if a linear activation function (y=x) was applied.
print(output.shape)
print(output)