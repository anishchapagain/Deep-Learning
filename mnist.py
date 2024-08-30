import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # train = True

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(mnist_pytorch, batch_size=1, shuffle=False) # shuffle = False

# Create a figure to display the images
plt.figure(figsize=(15, 3))

# Print the first few images
for i, (image, label) in enumerate(train_loader_pytorch):
    if i < 8:  # Print the first 8 samples
        plt.subplot(1, 8, i + 1)
        plt.imshow(image[0].squeeze(), cmap='gray')
        plt.title(f"Label: {label.item()}")
        plt.axis('off')
    
plt.tight_layout()
plt.show()mnist
