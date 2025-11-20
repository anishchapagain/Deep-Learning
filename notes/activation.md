# Activation Functions in Neural Networks

## Why Are Activation Functions Necessary?

Activation functions are a critical component of neural networks because they introduce **non-linearity** into the system.

Without an activation function, a neural network, no matter how many layers it has, would behave just like a single-layer linear regression model. Each layer would simply be performing a linear transformation (`output = weights * input + bias`). Stacking these linear transformations results in another linear transformation.

By introducing non-linearity, activation functions allow the network to learn much more complex patterns and relationships in the data, which is essential for tackling non-trivial tasks like image recognition, natural language processing, and more.

## How Are Activation Functions Used?

In a typical neuron, the activation function is applied to the weighted sum of the inputs plus the bias.

1.  **Linear Step:** The inputs (`x`) are multiplied by their corresponding weights (`w`), and the bias (`b`) is added. This is often called the *logit* or *pre-activation* output.
    `z = w * x + b`
2.  **Non-Linear Step:** The activation function (`f`) is applied to the logit `z` to produce the final output of the neuron.
    `output = f(z)`

---

## Types of Activation Functions

Here are some of the most common activation functions used in neural networks.

### 1. Sigmoid (Logistic)

The Sigmoid function takes any real-valued number and "squashes" it into a range between 0 and 1.

**Mathematical Representation:**
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

**Unicode Formula:**
σ(z) = 1 / (1 + e⁻ᶻ)

**Derivative:**
$$ \sigma'(z) = \sigma(z) (1 - \sigma(z)) $$

**Explanation:**
It produces an "S"-shaped curve. The output can be interpreted as a probability, making it a popular choice for the output layer in binary classification problems.

**Pros:**
*   **Smooth Gradient:** It has a smooth, non-zero derivative everywhere, preventing sudden jumps in output values.
*   **Clear Output:** The output between 0 and 1 is easy to interpret as a probability.

**Cons:**
*   **Vanishing Gradient Problem:** For very high or very low input values, the derivative of the function becomes extremely small (close to zero). During backpropagation, this can cause the gradients to "vanish," making it difficult for the network to learn.
*   **Not Zero-Centered:** The output is always positive. This can lead to slower convergence during training.
*   **Computationally Expensive:** The exponential function is computationally more expensive than simpler operations.

**PyTorch Code:**
```python
import torch

z = torch.tensor([-1.0, 0.0, 1.0])
output = torch.sigmoid(z)
# tensor([0.2689, 0.5000, 0.7311])
```

**NumPy Code:**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-1.0, 0.0, 1.0])
output = sigmoid(z)
# array([0.26894142, 0.5       , 0.73105858])
```

### 2. Hyperbolic Tangent (Tanh)

The Tanh function is similar to the sigmoid but squashes values into a range between -1 and 1.

**Mathematical Representation:**
$$ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$

**Unicode Formula:**
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

**Derivative:**
$$ \tanh'(z) = 1 - \tanh^2(z) $$

**Explanation:**
It is also "S"-shaped but is zero-centered, meaning its output can be positive or negative.

**Pros:**
*   **Zero-Centered Output:** The output being centered around zero often helps speed up convergence during training compared to the sigmoid function.
*   **Stronger Gradients:** The derivatives of tanh are steeper than those of sigmoid, which can lead to faster learning.

**Cons:**
*   **Vanishing Gradient Problem:** Like the sigmoid function, it still suffers from the vanishing gradient problem for extreme input values.

**PyTorch Code:**
```python
import torch

z = torch.tensor([-1.0, 0.0, 1.0])
output = torch.tanh(z)
# tensor([-0.7616,  0.0000,  0.7616])
```

**NumPy Code:**
```python
import numpy as np

def tanh(z):
    return np.tanh(z)

z = np.array([-1.0, 0.0, 1.0])
output = tanh(z)
# array([-0.76159416,  0.        ,  0.76159416])
```

### 3. Rectified Linear Unit (ReLU)

ReLU is one of the most widely used activation functions in deep learning.

**Mathematical Representation:**
$$ \text{ReLU}(z) = \max(0, z) $$

**Unicode Formula:**
ReLU(z) = max(0, z)

**Derivative:**
$$ \text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases} $$

**Explanation:**
It outputs the input directly if it is positive, and it outputs zero otherwise.

**Pros:**
*   **Computationally Efficient:** It is very simple to compute (just a `max(0, z)` operation).
*   **Avoids Vanishing Gradients (for positive inputs):** For positive inputs, the derivative is always 1, which helps prevent the vanishing gradient problem and allows models to learn faster.
*   **Sparsity:** Because it outputs zero for negative inputs, it can lead to "sparse" representations in the network, where some neurons are inactive. This can make the network more efficient.

**Cons:**
*   **The "Dying ReLU" Problem:** If a neuron's input is always negative, it will always output zero. As a result, the gradient for that neuron will also always be zero, and it will stop learning entirely (it "dies").
*   **Not Zero-Centered:** The outputs are always non-negative.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

z = torch.tensor([-1.0, 0.0, 1.0])

# Functional approach
output_functional = torch.relu(z)
# tensor([0., 0., 1.])

# Module approach
relu_module = nn.ReLU()
output_module = relu_module(z)
# tensor([0., 0., 1.])
```

**NumPy Code:**
```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

z = np.array([-1.0, 0.0, 1.0])
output = relu(z)
# array([0., 0., 1.])
```

### 4. Leaky ReLU

Leaky ReLU is a variant of ReLU designed to address the "Dying ReLU" problem.

**Mathematical Representation:**
$$ \text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases} $$
Where `α` is a small positive constant (e.g., 0.01).

**Unicode Formula:**
LeakyReLU(z) = z if z > 0 or αz if z ≤ 0

**Derivative:**
$$ \text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \le 0 \end{cases} $$

**Explanation:**
Instead of outputting zero for negative inputs, it outputs a small, non-zero, negative value. This ensures that the neuron never has a zero gradient and can continue to learn.

**Pros:**
*   **Fixes the "Dying ReLU" Problem:** It prevents neurons from becoming completely inactive.

**Cons:**
*   **Inconsistent Results:** Its performance is not always better than standard ReLU.
*   **Extra Hyperparameter:** The value of `α` is another hyperparameter that needs to be tuned.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

z = torch.tensor([-1.0, 0.0, 1.0])

# Functional approach
output_functional = torch.nn.functional.leaky_relu(z, negative_slope=0.01)
# tensor([-0.0100,  0.0000,  1.0000])

# Module approach
leaky_relu_module = nn.LeakyReLU(negative_slope=0.01)
output_module = leaky_relu_module(z)
# tensor([-0.0100,  0.0000,  1.0000])
```

**NumPy Code:**
```python
import numpy as np

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

z = np.array([-1.0, 0.0, 1.0])
output = leaky_relu(z)
# array([-0.01,  0.  ,  1.  ])
```

### 5. Parametric ReLU (PReLU)

PReLU is an extension of Leaky ReLU where the slope for negative inputs is learned during training.

**Mathematical Representation:**
$$ \text{PReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases} $$
Where `α` is a *learnable* parameter.

**Unicode Formula:**
PReLU(z) = z if z > 0 or αz if z ≤ 0 (where α is a learnable parameter)

**Derivative:**
$$ \text{PReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \le 0 \end{cases} $$
(where α is a learnable parameter)

**Explanation:**
Instead of having a fixed slope `α`, PReLU treats `α` as a parameter that the network learns through backpropagation. This can allow the network to find the optimal slope for its specific task.

**Pros:**
*   **Adaptive Learning:** Can potentially lead to better performance by learning the most appropriate slope for negative inputs.
*   **Fixes the "Dying ReLU" Problem:** Like Leaky ReLU, it prevents neurons from becoming completely inactive.

**Cons:**
*   **Risk of Overfitting:** On smaller datasets, learning the `α` parameter can lead to overfitting.
*   **More Parameters:** Adds a small number of extra parameters to the model.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

z = torch.tensor([-1.0, 0.0, 1.0])

# PReLU is typically used as a module
prelu_module = nn.PReLU()
output = prelu_module(z)
# The output will depend on the learned alpha value
```

**NumPy Code:**
```python
import numpy as np

# The implementation is the same as Leaky ReLU, the key difference is that
# the alpha value is a learnable parameter during model training.
def prelu(z, alpha):
    return np.where(z > 0, z, z * alpha)

# Example with a pre-determined alpha
z = np.array([-1.0, 0.0, 1.0])
alpha = 0.25 # This value would be learned by the model
output = prelu(z, alpha)
# array([-0.25,  0.  ,  1.  ])
```

### 6. Exponential Linear Unit (ELU)

ELU is another variant of ReLU that aims to be more robust and produce negative outputs.

**Mathematical Representation:**
$$ \text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha (e^z - 1) & \text{if } z \le 0 \end{cases} $$
Where `α` is a positive constant.

**Unicode Formula:**
ELU(z) = z if z > 0 or α(eᶻ - 1) if z ≤ 0

**Derivative:**
$$ \text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z & \text{if } z \le 0 \end{cases} $$

**Explanation:**
For positive inputs, it behaves like ReLU. For negative inputs, it has a smooth, saturating curve that approaches -α.

**Pros:**
*   **Produces Negative Outputs:** This can help push the mean of the activations closer to zero, which can speed up learning.
*   **Fixes the "Dying ReLU" Problem:** It has a non-zero gradient for negative values.
*   **Smooth Transition:** Unlike the sharp corner in ReLU and Leaky ReLU, ELU is smooth for negative inputs, which can improve robustness.

**Cons:**
*   **Computationally Expensive:** The presence of the exponential function makes it more computationally intensive than ReLU.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

z = torch.tensor([-1.0, 0.0, 1.0])

# Module approach
elu_module = nn.ELU(alpha=1.0)
output = elu_module(z)
# tensor([-0.6321,  0.0000,  1.0000])
```

**NumPy Code:**
```python
import numpy as np

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

z = np.array([-1.0, 0.0, 1.0])
output = elu(z)
# array([-0.63212056,  0.        ,  1.        ])
```

### 7. Softmax

Softmax is typically used in the final output layer of a multi-class classification network.

**Mathematical Representation:**
$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i=1, \dots, K $$

**Unicode Formula:**
Softmax(zᵢ) = eᶻᵢ / Σⱼ(eᶻⱼ)

**Derivative:**
The derivative of the softmax function is a bit more complex because it is a vector function. The derivative of the i-th output with respect to the j-th input is:
$$ \frac{\partial \text{Softmax}(z)_i}{\partial z_j} = \text{Softmax}(z)_i (\delta_{ij} - \text{Softmax}(z)_j) $$
Where `δᵢⱼ` is the Kronecker delta (it is 1 if `i=j` and 0 otherwise).

**Explanation:**
It takes a vector of K real-valued scores (logits) and converts them into a probability distribution over K classes. The sum of all the output probabilities will be 1.

**Pros:**
*   **Probability Distribution:** It produces a clear probability distribution over multiple classes, making it ideal for multi-class classification.

**Cons:**
*   **Output Layer Only:** It is almost exclusively used in the output layer.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

z = torch.tensor([1.0, 2.0, 3.0])

# Functional approach
output_functional = torch.softmax(z, dim=0)
# tensor([0.0900, 0.2447, 0.6652])

# Module approach
softmax_module = nn.Softmax(dim=0)
output_module = softmax_module(z)
# tensor([0.0900, 0.2447, 0.6652])
```

**NumPy Code:**
```python
import numpy as np

def softmax(z):
    # Subtracting the max for numerical stability is a common trick
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)

z = np.array([1.0, 2.0, 3.0])
output = softmax(z)
# array([0.09003057, 0.24472847, 0.66524096])
```

### 8. Linear Activation

The Linear activation function, also known as the "identity" function, is the simplest type of activation function. It does not apply any transformation to the input; the output is simply equal to the input.

**Mathematical Representation:**
$$ f(z) = z $$

**Unicode Formula:**
f(z) = z

**Explanation:**
It is a straight line, meaning the output is directly proportional to the input. While it might seem counterintuitive to have an "activation" that doesn't change the input, it is the default for the output layer in regression problems.

**Pros:**
*   **Simplicity:** It is computationally trivial.
*   **Direct Output for Regression:** It allows the output of a regression model to take on any real value, which is exactly what is needed when predicting continuous quantities.

**Cons:**
*   **No Non-Linearity:** If used in hidden layers, the entire network collapses into a single linear model, defeating the purpose of a deep neural network. It cannot learn complex patterns.
*   **Limited Use:** Its use is almost exclusively confined to the output layer of regression models.

**PyTorch Code:**
```python
import torch
import torch.nn as nn

# There is no specific 'linear' activation function layer in PyTorch
# because it's the default behavior. A linear layer's output is not
# passed through a non-linearity.
z = torch.tensor([-1.0, 0.0, 1.0])
output = z # No activation applied
# tensor([-1.,  0.,  1.])
```

**NumPy Code:**
```python
import numpy as np

def linear(z):
    return z

z = np.array([-1.0, 0.0, 1.0])
output = linear(z)
# array([-1.,  0.,  1.])
```

### 9. Binary Step Function

The Binary Step function is a simple threshold-based activation function. If the input value is above a certain threshold, it outputs one value (e.g., 1), and if it is below the threshold, it outputs another (e.g., 0).

**Mathematical Representation:**
$$ f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases} $$

**Unicode Formula:**
f(z) = 1 if z ≥ 0 or 0 if z < 0

**Explanation:**
It is a very simple function that "activates" a neuron in a binary fashion. It was historically important in the early days of neural networks (like the Perceptron) but is rarely used in modern deep learning models.

**Pros:**
*   **Simple and Intuitive:** Very easy to understand and implement.
*   **Clear Binary Output:** Useful for binary classification tasks where a definite decision is required.

**Cons:**
*   **Zero Gradient:** The derivative of the function is zero almost everywhere (and undefined at the threshold). This makes it impossible to use with gradient-based optimization methods like backpropagation, as no learning can occur.
*   **Not Informative:** It only provides a binary output and does not convey any information about the "confidence" of the activation. For example, an input of 0.1 and an input of 100 both produce the same output of 1.

**PyTorch Code:**
```python
import torch

# PyTorch does not have a built-in binary step function for automatic
# differentiation because its gradient is zero everywhere.
# However, it can be implemented manually.
def binary_step(z):
    return (z >= 0).float()

z = torch.tensor([-1.0, 0.0, 1.0])
output = binary_step(z)
# tensor([0., 1., 1.])
```

**NumPy Code:**
```python
import numpy as np

def binary_step(z):
    return np.where(z >= 0, 1, 0)

z = np.array([-1.0, 0.0, 1.0])
output = binary_step(z)
# array([0, 1, 1])
```

---

## Choosing the Right Activation Function

### For Hidden Layers

*   **Start with ReLU:** It is the most common choice due to its computational efficiency and good performance in most scenarios.
*   **If you experience the "Dying ReLU" problem,** consider switching to one of its variants:
    *   **Leaky ReLU**
    *   **PReLU** (Parametric ReLU)
    *   **ELU** (Exponential Linear Unit)
*   **Tanh** can also be effective, especially in recurrent neural networks (RNNs), but ReLU and its variants are more common in convolutional neural networks (CNNs) and feed-forward networks.
*   **Avoid Sigmoid:** The Sigmoid function is generally not recommended for hidden layers due to its tendency to cause vanishing gradients.

### For Output Layers

The choice for the output layer is determined by the type of problem you are solving:

*   **Regression:** For tasks where you need to predict a continuous value (e.g., the price of a house), use **no activation function** (i.e., a linear activation). The raw output from the final layer will be your prediction.

*   **Binary Classification:** For tasks with two possible outcomes (e.g., cat or dog), use the **Sigmoid** function. It will output a value between 0 and 1, which can be interpreted as the probability of the positive class.

*   **Multi-Class Classification:** For tasks with more than two possible outcomes (e.g., classifying an image as a cat, dog, or bird), use the **Softmax** function. It will output a probability distribution across all classes, with the probabilities summing to 1.

---

## Characteristics of a Good Activation Function

When choosing or designing an activation function, several characteristics are desirable for optimal neural network performance:

*   **Non-linearity:** This is the most fundamental characteristic. Without non-linearity, a neural network, regardless of its depth, would only be able to learn linear transformations, severely limiting its ability to model complex data. (e.g., Sigmoid, Tanh, ReLU, ELU)
*   **Differentiability:** For gradient-based optimization algorithms like backpropagation to work, the activation function must be differentiable across its domain. This allows for the computation of gradients needed to update network weights. (e.g., Sigmoid, Tanh, ReLU, ELU - Binary Step is not differentiable)
*   **Monotonicity:** A monotonic activation function (either always increasing or always decreasing) ensures that the network's output changes consistently with the input. This helps in the convergence of gradient descent. (e.g., Sigmoid, Tanh, ReLU, ELU)
*   **Approximation to Identity (for hidden layers):** When inputs are small, an activation function that behaves like the identity function (output ≈ input) can help mitigate the vanishing gradient problem, allowing gradients to flow more effectively through the network. (e.g., ELU, Leaky ReLU, PReLU)
*   **Zero-centered Output:** If the output of the activation function is centered around zero (i.e., has both positive and negative values), it can help speed up convergence during training. This is because it allows gradients to update weights in both positive and negative directions more efficiently. (e.g., Tanh, ELU)
*   **Computational Efficiency:** Activation functions are applied to every neuron in every layer. Therefore, a computationally inexpensive function can significantly reduce training and inference times. (e.g., ReLU, Leaky ReLU, PReLU)
*   **Avoidance of Vanishing/Exploding Gradients:** This is crucial for training deep neural networks. Functions that saturate (gradients become very small) or grow too rapidly (gradients become very large) can hinder learning. Modern activation functions like ReLU and its variants are designed to address these issues. (e.g., ReLU, Leaky ReLU, PReLU, ELU)

---

## Unicode Math Formulas for PowerPoint

*   **Sigmoid:** σ(z) = 1 / (1 + e⁻ᶻ)
*   **Tanh:** tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
*   **ReLU:** ReLU(z) = max(0, z)
*   **Leaky ReLU:** LeakyReLU(z) = z if z > 0 or αz if z ≤ 0
*   **PReLU:** PReLU(z) = z if z > 0 or αz if z ≤ 0 (where α is a learnable parameter)
*   **ELU:** ELU(z) = z if z > 0 or α(eᶻ - 1) if z ≤ 0
*   **Softmax:** Softmax(zᵢ) = eᶻᵢ / Σⱼ(eᶻⱼ)

---

## Mathematical Symbols

| Name      | Symbol |
| :-------- | :----: |
| Sigma     |   σ    |
| Alpha     |   α    |
| Delta     |   δ    |
| Summation |   Σ    |