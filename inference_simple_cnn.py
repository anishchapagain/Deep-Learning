# inference_simple_cnn.py
"""
Inference script for the SimpleCNN model defined in cnn_mnist.ipynb.
It loads the model architecture, optionally loads a saved state_dict, preprocesses a new image,
and runs inference to predict the class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os

# Define the SimpleCNN model (copied from cnn_mnist.ipynb)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W (28 -> 14 -> 7)
        # Fully‑connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 * 7 * 7 is the output of the last conv layer
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(state_path: str = None, device: str = "cpu") -> SimpleCNN:
    """Instantiate the model and optionally load a saved state_dict.
    Args:
        state_path: Path to a .pth file containing the model's state_dict.
        device: Device to map the model to ("cpu" or "cuda").
    Returns:
        An instance of SimpleCNN ready for inference.
    """
    model = SimpleCNN().to(device)
    if state_path:
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"State dict not found at {state_path}")
        state_dict = torch.load(state_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocessing for a single MNIST image (grayscale 28x28)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

def predict_image(model: SimpleCNN, image_path: str, device: str = "cpu"):
    """Run inference on a single image and return the predicted class index.
    Args:
        model: Loaded SimpleCNN model.
        image_path: Path to the image file.
        device: Device for computation.
    Returns:
        Predicted class index (int).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = Image.open(image_path).convert("L")  # Ensure grayscale
    img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

def main():
    parser = argparse.ArgumentParser(description="SimpleCNN inference on a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--state_path", type=str, default=None, help="Optional .pth file with trained weights.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda).")
    args = parser.parse_args()

    model = load_model(state_path=args.state_path, device=args.device)
    pred = predict_image(model, args.image_path, device=args.device)
    print(f"Predicted class: {pred}")

if __name__ == "__main__":
    main()
