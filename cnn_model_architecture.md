
# Convolutional Neural Network Architectures

This document provides a detailed overview of three influential Convolutional Neural Network (CNN) architectures: LeNet-5, VGG-16, and GoogLeNet.

## LeNet-5

### Purpose

LeNet-5 is one of the earliest and most foundational CNN architectures, developed by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner in 1998. Its primary purpose was for **handwritten and machine-printed character recognition**. It was famously used by banks to read checks.

### Architecture

![LeNet-5 Architecture](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhJPuyrK8blPR7kM7B9e_eWaLEkl6_8XQLWQ&s)

LeNet-5 has a simple and straightforward architecture, consisting of 7 layers (excluding the input layer). It's designed to work with grayscale images of size 32x32.

1.  **C1 - Convolutional Layer:** 6 filters of size 5x5 with a stride of 1. Output: 28x28x6.
2.  **S2 - Subsampling (Average Pooling) Layer:** Filter size 2x2 with a stride of 2. Output: 14x14x6.
3.  **C3 - Convolutional Layer:** 16 filters of size 5x5 with a stride of 1. Output: 10x10x16.
4.  **S4 - Subsampling (Average Pooling) Layer:** Filter size 2x2 with a stride of 2. Output: 5x5x16.
5.  **C5 - Convolutional Layer (Fully Connected):** 120 filters of size 5x5. This is equivalent to a fully connected layer. Output: 1x1x120.
6.  **F6 - Fully Connected Layer:** 84 units.
7.  **Output Layer:** 10 units (for 10-digit classification) with a softmax activation function.

### Pros and Cons

**Pros:**

*   **Pioneering:** It was one of the first CNNs and laid the groundwork for modern deep learning.
*   **Simple:** The architecture is easy to understand and implement.
*   **Efficient:** For its time, it was computationally efficient.

**Cons:**

*   **Shallow:** Compared to modern architectures, it's very shallow, limiting its ability to learn complex features.
*   **Small Input Size:** Designed for small 32x32 grayscale images.
*   **Not as powerful:** Modern architectures are significantly more powerful.

### Practical Lab Tasks

*   **Implement from scratch:** Build the LeNet-5 architecture using a deep learning framework like PyTorch or TensorFlow.
*   **Train on MNIST:** Train the model on the MNIST dataset and try to replicate the original paper's results.
*   **Visualize Filters:** Visualize the learned filters in the convolutional layers to understand what features the network is learning.

### When to Use

LeNet-5 is now mostly of historical and educational importance. It's a great architecture to start with to understand the basic building blocks of CNNs. It's suitable for simple, small-scale image classification tasks, especially with grayscale images, like digit recognition.

### Practical Hands-on Cases

*   **MNIST Handwritten Digit Recognition:** The classic use case for LeNet-5.
*   **Character Recognition:** Recognizing characters from simple documents.
*   **Basic Object Recognition:** On small, low-resolution images.

### RGB Support

The original LeNet-5 was designed for **grayscale images**. To use it with RGB images, you would need to either convert the images to grayscale or modify the first convolutional layer to accept 3 input channels instead of 1.

### PyTorch Example

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage:
# model = LeNet5()
# input_tensor = torch.randn(1, 1, 32, 32) # (batch_size, channels, height, width)
# output = model(input_tensor)
# print(output.shape)
```

## VGG-16

### Purpose

VGG-16, developed by the Visual Geometry Group at the University of Oxford, was a significant step forward in CNN design. Its main contribution was demonstrating that **depth is a critical component for good performance**. It was a top performer in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2014.

### Architecture

![VGG-16 Architecture](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

VGG-16 is known for its simplicity and uniformity. It uses a consistent 3x3 filter size for all convolutional layers and 2x2 for all pooling layers. The "16" in its name refers to the 16 layers with learnable weights.

The architecture consists of a stack of convolutional layers followed by three fully connected layers.

*   **Input:** 224x224 RGB image.
*   **Convolutional Blocks:**
    *   2 x Conv3-64 (3x3 filters, 64 channels) followed by MaxPool
    *   2 x Conv3-128 followed by MaxPool
    *   3 x Conv3-256 followed by MaxPool
    *   3 x Conv3-512 followed by MaxPool
    *   3 x Conv3-512 followed by MaxPool
*   **Fully Connected Layers:**
    *   FC-4096
    *   FC-4096
    *   FC-1000 (for 1000 classes in ImageNet) with Softmax

### Pros and Cons

**Pros:**

*   **Simple and Uniform:** The architecture is easy to understand and implement due to its consistent use of 3x3 convolutions and 2x2 pooling.
*   **Good for Transfer Learning:** It is a great feature extractor and is widely used for transfer learning.
*   **Good Performance:** It achieved state-of-the-art performance on ImageNet in 2014.

**Cons:**

*   **Computationally Expensive:** It has a large number of parameters (138M) and is computationally expensive to train.
*   **Large Memory Footprint:** The large number of parameters also means it has a large memory footprint.

### Practical Lab Tasks

*   **Transfer Learning for Image Classification:** Use a pre-trained VGG-16 model and fine-tune it for a custom image classification task (e.g., classifying different types of flowers or animals).
*   **Feature Extraction for Object Detection:** Use the convolutional base of VGG-16 as a feature extractor for an object detection model like Faster R-CNN.
*   **Style Transfer:** Use the feature maps from different layers of VGG-16 to perform neural style transfer.

### When to Use

VGG-16 is a powerful and versatile model. It's a great choice when you need a robust feature extractor. It's often used as a backbone for more complex tasks like object detection and semantic segmentation. However, it is a large model with a high number of parameters (around 138 million), making it computationally expensive.

### Practical Hands-on Cases

*   **Image Classification:** For a wide variety of image classification tasks.
*   **Transfer Learning:** VGG-16 is a very popular choice for transfer learning. You can use the pre-trained convolutional base to extract features from your images and then train a smaller classifier on top of it for your specific task.
*   **Object Detection and Segmentation:** The feature maps from VGG-16 can be used as input to object detection or segmentation models.

### RGB Support

Yes, VGG-16 is designed to work with **RGB images** (3 channels).

### PyTorch Example

```python
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Example usage:
# model = VGG16()
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)
```

## GoogLeNet (Inception v1)

### Purpose

GoogLeNet, also known as Inception v1, was the winner of the ILSVRC 2014. Its main innovation was the **Inception module**, which aimed to be more computationally efficient than VGG-style networks while achieving state-of-the-art performance. It introduced the idea of using different filter sizes in parallel to capture features at multiple scales.

### Architecture

![GoogLeNet Architecture](https://viso.ai/wp-content/uploads/2024/04/google-net-like-1280x368.png)

GoogLeNet's architecture is 22 layers deep. The core component is the Inception module.

**Inception Module:** An Inception module has multiple "branches" of convolutions with different filter sizes (1x1, 3x3, 5x5) and a max-pooling layer. The outputs of these branches are concatenated together. A key feature is the use of **1x1 convolutions** for dimensionality reduction before the more expensive 3x3 and 5x5 convolutions.

**Global Average Pooling:** Instead of large fully connected layers at the end, GoogLeNet uses global average pooling, which averages each feature map to a single number. This drastically reduces the number of parameters.

**Auxiliary Classifiers:** The network has two auxiliary classifiers in the middle of the network. These were used during training to combat the vanishing gradient problem.

### Pros and Cons

**Pros:**

*   **Computationally Efficient:** It is much more computationally efficient than VGG-16, with fewer parameters (4M vs 138M).
*   **Good Performance:** It won the ILSVRC 2014 competition.
*   **Multi-scale Feature Extraction:** The Inception module allows the network to learn features at multiple scales.

**Cons:**

*   **Complex Architecture:** The Inception module is more complex to implement than the simple convolutional layers of VGG-16.

### Practical Lab Tasks

*   **Implement the Inception Module:** Implement the Inception module from scratch and use it to build a smaller version of GoogLeNet.
*   **Train on CIFAR-10:** Train the model on the CIFAR-10 dataset and compare its performance to LeNet-5 and VGG-16.
*   **Investigate the effect of 1x1 convolutions:** Train a version of the Inception module without the 1x1 convolutions and compare its performance and computational cost to the original version.

### When to Use

GoogLeNet is a good choice when you need a deep and powerful network but are constrained by computational resources. It's more efficient than VGG-16 in terms of both parameters and computation. It's a good starting point for tasks where you need to capture features at different scales.

### Practical Hands-on Cases

*   **Image Classification:** Excellent for image classification tasks, especially when computational efficiency is a concern.
*   **Real-time Image Recognition:** Due to its efficiency, it can be deployed in applications where real-time performance is important.
*   **Transfer Learning:** The pre-trained GoogLeNet can be used for transfer learning, similar to VGG-16.

### RGB Support

Yes, GoogLeNet is designed to work with **RGB images** (3 channels), with an input size of 224x224.

### PyTorch Example

```python
import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Example usage:
# model = GoogLeNet()
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)
```

## References

*   **LeNet-5:**
    *   [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (Original Paper)
*   **VGG-16:**
    *   [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Original Paper)
*   **GoogLeNet (Inception v1):**
    *   [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (Original Paper)
