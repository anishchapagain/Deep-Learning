# Deep Learning & Neural Networks: A Comprehensive Overview

This document serves as a high-level summary and tutorial on Deep Learning, aggregating key concepts from various notes in this repository. It covers the foundations of neural networks, advanced architectures, training techniques, and coding standards.

## 1. Neural Network Fundamentals

### 1.1. Activation Functions
Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns.
*   **Sigmoid**: Maps input to (0, 1). Used in binary classification. Susceptible to vanishing gradients.
*   **Tanh**: Maps input to (-1, 1). Zero-centered but also suffers from vanishing gradients.
*   **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$. Most popular, solves vanishing gradient for positive values, but suffers from "dying ReLU".
*   **Leaky ReLU / PReLU**: Allows a small gradient for negative values to fix dying ReLU.
*   **Softmax**: Converts logits into probabilities summing to 1. Used for multi-class classification.

### 1.2. Loss Functions
Loss functions quantify the difference between predicted and actual values.
*   **Mean Squared Error (MSE)**: For regression tasks.
*   **Binary Cross-Entropy (BCE)**: For binary classification.
*   **Cross-Entropy Loss**: For multi-class classification.

### 1.3. Optimization & Gradient Descent
Optimization algorithms adjust model parameters to minimize the loss function.
*   **Gradient Descent**: Iteratively moves towards the minimum of the loss function using the gradient.
    *   **Batch GD**: Updates weights using the entire dataset. Stable but slow.
    *   **Stochastic GD (SGD)**: Updates weights using one sample. Fast but noisy.
    *   **Mini-Batch GD**: Updates weights using a batch of samples. Best of both worlds.
*   **Backpropagation**: The algorithm to calculate gradients via the chain rule.
*   **Optimizers**:
    *   **Momentum**: Accelerates SGD in the relevant direction and dampens oscillations.
    *   **Adam**: Adaptive Moment Estimation. Combines Momentum and RMSprop. widely used default.

### 1.4. Regularization & Initialization
Techniques to improve generalization and training stability.
*   **Regularization**: Prevents overfitting.
    *   **L1/L2 Regularization**: Adds a penalty term to the loss function based on weight magnitude.
    *   **Dropout**: Randomly deactivates neurons during training to force redundant feature learning.
    *   **Early Stopping**: Stops training when validation loss stops improving.
*   **Batch Normalization**: Normalizes layer inputs to stabilize learning and reduce internal covariate shift.
*   **Initialization**: Setting initial weights properly (e.g., Xavier/Glorot, He) is crucial to avoid vanishing/exploding gradients.

## 2. Deep Learning Architectures

### 2.1. Multi-layer Perceptrons (MLP)
The simplest feedforward neural network consisting of an input layer, one or more hidden layers, and an output layer. Good for tabular data but struggles with spatial/sequential data.

### 2.2. Convolutional Neural Networks (CNN)
Designed for grid-like data (images).
*   **Components**: Convolutional layers (feature extraction), Pooling layers (downsampling), Fully Connected layers (classification).
*   **Architectures**:
    *   **LeNet-5**: Early CNN for digit recognition.
    *   **AlexNet**: Deep CNN that popularized deep learning.
    *   **VGG**: Very deep network with small filters.
    *   **ResNet**: Introduces residual connections (skip connections) to train very deep networks.

### 2.3. Recurrent Neural Networks (RNN)
Designed for sequential data (time series, text).
*   **Vanilla RNN**: Maintains a hidden state to remember previous inputs. Suffers from vanishing gradients on long sequences.
*   **LSTM (Long Short-Term Memory)**: Uses gates (input, forget, output) to control information flow and handle long-term dependencies.
*   **GRU (Gated Recurrent Unit)**: Simplified LSTM with update and reset gates.

### 2.4. Transformers
State-of-the-art for NLP and increasingly Vision.
*   **Mechanism**: Relies on **Self-Attention** to weigh the importance of different parts of the input sequence, allowing parallel processing and handling long-range dependencies better than RNNs.

### 2.5. Generative Models
*   **Autoencoders**: Unsupervised models that learn to compress (encode) and reconstruct (decode) data. Used for dimensionality reduction, denoising, and generation.
*   **GANs (Generative Adversarial Networks)**: Consist of a **Generator** (creates fake data) and a **Discriminator** (distinguishes real vs. fake). They compete in a minimax game to produce realistic data.

## 3. Training & Practical Techniques

### 3.1. Data Augmentation
Technique to artificially increase the size and diversity of the training set.
*   **Images**: Flipping, rotation, cropping, color jittering.
*   **Text**: Synonym replacement, random insertion/deletion.

### 3.2. Transfer Learning
Reusing a pre-trained model on a new, related task.
*   **Feature Extraction**: Freezing the base network and training a new classifier.
*   **Fine-tuning**: Unfreezing some or all layers of the base network and training with a low learning rate.

### 3.3. Model Loading & Saving
*   **State Dict**: A Python dictionary mapping each layer to its parameter tensor.
*   **Saving**: `torch.save(model.state_dict(), PATH)`
*   **Loading**: `model.load_state_dict(torch.load(PATH))`
*   **Scenarios**: Resuming training, inference, transfer learning.

### 3.4. Hyperparameters
Parameters set before training (not learned).
*   **Examples**: Learning rate, batch size, number of epochs, optimizer choice, network architecture (layers, units).
*   **Tuning**: Grid search, random search, Bayesian optimization.

## 4. Coding Standards
Adhering to standards ensures code quality and reproducibility.
*   **Python**: PEP 8, type hinting, docstrings.
*   **Project Structure**: Organized directories for data, models, notebooks, scripts.
*   **MLOps**: Experiment tracking, model versioning, reproducibility.

---
*This overview synthesizes content from the `notes` directory. Refer to individual files for detailed explanations and code examples.*
