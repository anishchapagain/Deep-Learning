# Deep Learning Fundamentals – Question Bank

## Short‑Answer Questions
1. **What is a biological neuron and how does it inspire artificial neurons?**
2. **Define a Multilayer Perceptron (MLP) and its typical architecture.**
3. **What is the purpose of an activation function in a neural network?**
4. **State the difference between a forward pass and a backward pass.**
5. **Name three commonly used activation functions and one key property of each.**
6. **What is back‑propagation and why is it essential for training neural networks?**
7. **List two gradient‑based optimizers and one advantage of each over plain Stochastic Gradient Descent (SGD).**

## Long‑Answer / Essay Questions
8. **Explain the complete forward‑pass computation in a simple three‑layer MLP (input, hidden, output). Include the role of weights, biases, and activation functions.**
9. **Derive the back‑propagation update rule for a single weight in a network with a mean‑squared‑error loss. Show each step from loss gradient to weight adjustment.**
10. **Compare and contrast the following optimizers: SGD with momentum, RMSProp, and Adam. Discuss how each algorithm adapts the learning rate and the impact on convergence speed and stability.**
11. **Discuss the trade‑offs involved in choosing an activation function for deep networks (e.g., ReLU vs. sigmoid vs. tanh). Include considerations of vanishing/exploding gradients and computational efficiency.**
12. **Design a short quiz (5 questions) that tests understanding of forward pass, backward pass, and optimizer selection. Provide the correct answer key separately.**

## Complex Exam Questions (Marks Distribution)
### 1‑Mark Questions (10)
1. What does the abbreviation **MLP** stand for?
2. Write the mathematical expression for a single neuron output using a linear activation.
3. Define **epoch** in the context of training neural networks.
4. State the purpose of a **bias term** in a neuron.
5. Name one advantage of using **ReLU** over **sigmoid**.
6. What is the shape of the weight matrix for a layer connecting 128 inputs to 64 neurons?
7. Write the formula for the **mean squared error (MSE)** loss.
8. Which optimizer adapts learning rates per parameter using first‑ and second‑moment estimates?
9. What does **back‑propagation** compute?
10. In LaTeX, how do you write the gradient symbol \(\nabla\)?

### 2‑Mark Questions (15)
11. Explain why **vanishing gradients** hinder training of deep networks.
12. Derive the gradient of the MSE loss with respect to the network output \(\hat{y}\).
13. Describe the role of **momentum** in SGD.
14. Write the update rule for **Adam** including bias‑correction terms.
15. Compare **tanh** and **sigmoid** activation functions in terms of output range and centering.
16. Explain how **weight decay** (L2 regularization) modifies the loss function.
17. Provide the formula for the **softmax** function for a vector \(z\).
18. What is the effect of **learning rate decay** on training dynamics?
19. State the difference between **batch** and **mini‑batch** gradient descent.
20. Write the expression for the **cross‑entropy** loss for binary classification.
21. Explain the concept of **over‑parameterization** in neural networks.
22. How does **dropout** help prevent overfitting?
23. Describe the **chain rule** as used in back‑propagation.
24. What does **early stopping** monitor during training?
25. Provide the formula for the **L1 regularization** term.

### 5‑Mark Questions (15)
26. Derive the back‑propagation equations for a two‑layer MLP with sigmoid activations, showing gradients for both layers.
27. Given a network with ReLU activations, explain why gradients can become zero and how **Leaky ReLU** mitigates this.
28. Compute the gradient of the cross‑entropy loss combined with softmax for a multi‑class output.
29. Show step‑by‑step how **RMSProp** updates the learning rate for a single parameter.
30. Derive the update rule for **SGD with Nesterov momentum**.
31. Explain the effect of **batch normalization** on the forward and backward passes.
32. Provide a detailed derivation of the **AdamW** optimizer and contrast it with standard Adam.
33. For a convolutional layer with kernel size \(3\times3\), stride 1, and padding 1, compute the output dimensions given an input of \(32\times32\) with 16 channels.
34. Derive the loss gradient for a **hinge loss** used in SVM‑style classification.
35. Explain how **gradient clipping** prevents exploding gradients, and provide the clipping formula.
36. Provide a full derivation of the **learning rate schedule** for cosine annealing.
37. Derive the back‑propagation update for a **batch‑norm** layer, including the scale and shift parameters.
38. Explain the mathematical intuition behind **weight initialization** (e.g., He vs. Xavier).
39. Derive the expression for the **Fisher Information Matrix** in the context of neural network parameters.
40. Show how to compute the **Jacobian matrix** of a network’s output with respect to its inputs.

### 10‑Mark Questions (10)
41. Design a complete three‑layer neural network (input, hidden, output) for MNIST digit classification, specifying architecture, activation functions, loss, and optimizer. Provide full forward‑pass equations and a complete back‑propagation derivation for all parameters.
42. Compare and critically evaluate **SGD**, **Adam**, **AdaGrad**, and **RMSProp** on convergence speed, stability, and generalization, supporting your discussion with mathematical analysis and empirical considerations.
43. Derive the **second‑order optimization** update using the Hessian matrix and discuss why it is rarely used in deep learning.
44. Provide a thorough explanation of **transformer attention mechanisms**, including the scaled dot‑product attention formula and its gradient computation.
45. Formulate the **variational auto‑encoder (VAE)** loss function, derive the KL‑divergence term, and explain back‑propagation through the reparameterization trick.
46. Derive the **policy gradient** update rule for reinforcement learning, including the role of the advantage function.
47. Explain the mathematics behind **generative adversarial networks (GANs)**, derive the discriminator and generator loss gradients, and discuss mode collapse mitigation techniques.
48. Provide a detailed derivation of **layer‑wise relevance propagation (LRP)** for interpreting neural network predictions.
49. Derive the **optimal learning rate** for a quadratic loss surface using eigenvalue analysis.
50. Discuss the theoretical foundations of **neural tangent kernels (NTK)** and derive the kernel expression for a simple fully‑connected network.


## Computational Challenge Questions (Parameter Counting & Calculations)

### FFNN Parameter Counting (2-Mark Questions)
51. Calculate the total number of trainable parameters in a fully‑connected network with architecture: Input(784) → Hidden1(256) → Hidden2(128) → Output(10). Include biases.
52. A 3‑layer FFNN has 100,000 trainable parameters. If the input is 512‑dimensional and output is 5‑dimensional with one hidden layer of size \(h\), calculate \(h\).
53. Compare the parameter count of two architectures: (A) 784→512→10 vs (B) 784→256→256→10. Which has fewer parameters and by how much?

### CNN Parameter Counting (5-Mark Questions)
54. Calculate the total number of parameters in a convolutional layer with: 32 input channels, 64 output channels, kernel size \(5\times5\), and biases included.
55. For a CNN with the following architecture, compute total trainable parameters:
    - Conv1: 3 input channels, 16 filters, \(3\times3\) kernel
    - Conv2: 16 input channels, 32 filters, \(3\times3\) kernel
    - FC1: flattened → 128 units
    - FC2: 128 → 10 units
    - Input image: \(32\times32\times3\), after Conv layers the spatial size is \(8\times8\)
56. A depthwise separable convolution is used instead of standard convolution. Calculate the parameter reduction ratio for: 64 input channels, 64 output channels, \(3\times3\) kernel.

### Dimension Calculation (2-Mark Questions)
57. Given input \(224\times224\times3\), apply Conv(\(7\times7\), stride=2, padding=3, 64 filters) followed by MaxPool(\(3\times3\), stride=2). What is the output dimension?
58. Calculate output dimensions after: Input \(28\times28\times1\) → Conv(\(5\times5\), 32 filters, stride=1, no padding) → ReLU → MaxPool(\(2\times2\), stride=2).

### Gradient Computation with Numbers (5-Mark Questions)
59. Given a 2‑layer network with weights \(W^{(1)} = \begin{bmatrix}0.5 & -0.3\\0.2 & 0.8\end{bmatrix}\), \(W^{(2)} = \begin{bmatrix}0.4\\-0.6\end{bmatrix}\), input \(x = \begin{bmatrix}1\\2\end{bmatrix}\), sigmoid activations, and MSE loss with target \(y=0\), compute \(\frac{\partial L}{\partial W^{(1)}}\) numerically.
60. For a single convolutional filter \(W = \begin{bmatrix}1 & 0\\-1 & 2\end{bmatrix}\) applied to input \(X = \begin{bmatrix}3 & 1\\2 & 4\end{bmatrix}\) with stride=1, valid padding, compute the output and gradient \(\frac{\partial L}{\partial W}\) if \(L = \|output - target\|^2\) where \(target = \begin{bmatrix}5\end{bmatrix}\).

### Memory and FLOP Calculations (5-Mark Questions)
61. Calculate the memory required (in MB) to store activations for a batch size of 32 through the forward pass of: Input(\(224\times224\times3\)) → Conv(64 filters, \(7\times7\)) → Conv(128 filters, \(3\times3\)) → FC(1000). Assume 32‑bit floats.
62. Compute the number of FLOPs (floating‑point operations) for a single forward pass through a convolutional layer: input \(56\times56\times64\), output \(56\times56\times128\), kernel \(3\times3\), with biases.
63. Compare FLOPs between: (A) standard \(3\times3\) convolution with 128 input and 128 output channels on \(56\times56\) feature map, vs (B) \(1\times1\) bottleneck reducing to 32 channels, then \(3\times3\) conv, then \(1\times1\) expanding to 128 channels.

### Advanced Architecture Analysis (10-Mark Questions)
64. Design a CNN for \(32\times32\times3\) input to 10‑class output with exactly 50,000 parameters (±1000). Specify all layer dimensions, calculate actual parameter count, and verify.
65. Given a ResNet‑style block with skip connection: Input \(56\times56\times64\) → Conv(\(1\times1\), 64→16) → Conv(\(3\times3\), 16→16) → Conv(\(1\times1\), 16→64) + Input. Calculate: (a) total parameters, (b) FLOPs, (c) memory for batch=16, (d) gradient flow advantages.
66. For a FFNN with input 1024, hidden layers [512, 256, 128], output 10, calculate: (a) total parameters with/without biases, (b) memory to store all parameters in float32, (c) parameter ratio between largest and smallest layers, (d) if using dropout(0.5) in training, expected active neurons in each layer.

### Backpropagation Numerical Problems (5-Mark Questions)
67. Implement and compute one complete forward‑backward pass for XOR problem: inputs \(\{(0,0), (0,1), (1,0), (1,1)\}\), targets \(\{0,1,1,0\}\), 2→2→1 network, sigmoid activation, learning rate 0.1. Show weight updates for first sample.
68. Given a mini‑batch of size 4, compute batch‑normalized output and gradients for: input \(X = [1, 3, 5, 7]^T\), \(\gamma=2\), \(\beta=1\), \(\epsilon=0.001\). Then compute \(\frac{\partial L}{\partial X}\), \(\frac{\partial L}{\partial \gamma}\), \(\frac{\partial L}{\partial \beta}\) if upstream gradient is all ones.

### Multi‑Step Calculations (10-Mark Questions)
69. Design a complete architecture from scratch: Classify CIFAR‑10 (\(32\times32\times3\)) using max 1M parameters. Your design must include: (a) architecture specification, (b) parameter count verification, (c) output dimension tracking through all layers, (d) receptive field calculation, (e) memory requirement for batch=64.
70. For the VGG‑16 first block: Input(\(224\times224\times3\)) → Conv(\(3\times3\), 64) → Conv(\(3\times3\), 64) → MaxPool(\(2\times2\), stride=2). Calculate: (a) parameters per layer, (b) total parameters, (c) output dimensions after each operation, (d) FLOPs for entire block, (e) activation memory for batch=8, (f) gradient memory requirements.