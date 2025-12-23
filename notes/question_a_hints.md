# Deep Learning Fundamentals – Questions with Answers

## Short‑Answer Questions & Answers
1. **What is a biological neuron and how does it inspire artificial neurons?**
   - A biological neuron receives inputs through dendrites, processes them in the soma, and generates an output signal via the axon. Artificial neurons mimic this by computing a weighted sum of inputs, adding a bias, and applying an activation function to produce an output.

2. **Define a Multilayer Perceptron (MLP) and its typical architecture.**
   - An MLP is a feed‑forward neural network consisting of an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next, and non‑linear activation functions are applied between layers.

3. **What is the purpose of an activation function in a neural network?**
   - Activation functions introduce non‑linearity, enabling the network to model complex relationships beyond linear mappings.

4. **State the difference between a forward pass and a backward pass.**
   - *Forward pass*: computes the network output given inputs by propagating data layer‑by‑layer.
   - *Backward pass*: propagates the loss gradient from the output back through the network to compute gradients for each parameter (back‑propagation).

5. **Name three commonly used activation functions and one key property of each.**
   - **ReLU** – outputs zero for negative inputs, helping mitigate vanishing gradients.
   - **Sigmoid** – maps inputs to (0, 1), useful for probability outputs but prone to saturation.
   - **Tanh** – maps inputs to (‑1, 1), zero‑centered but also suffers from saturation.

6. **What is back‑propagation and why is it essential for training neural networks?**
   - Back‑propagation is the algorithm that computes gradients of the loss with respect to each weight using the chain rule. It is essential because it provides the information needed for gradient‑based optimizers to update the parameters.

7. **List two gradient‑based optimizers and one advantage of each over plain Stochastic Gradient Descent (SGD).**
   - **SGD with Momentum** – accelerates convergence by accumulating a velocity vector, reducing oscillations.
   - **Adam** – adapts learning rates per parameter using first‑ and second‑moment estimates, often converging faster with less tuning.

## Long‑Answer / Essay Questions & Sample Answers
8. **Explain the complete forward‑pass computation in a simple three‑layer MLP (input, hidden, output). Include the role of weights, biases, and activation functions.**
   - The input vector \(x\) is multiplied by the weight matrix \(W^{(1)}\) of the first layer and added to a bias vector \(b^{(1)}\) to produce \(z^{(1)} = W^{(1)}x + b^{(1)}\). An activation function \(\phi\) (e.g., ReLU) is applied: \(a^{(1)} = \phi(z^{(1)})\). This hidden activation \(a^{(1)}\) is then fed into the second layer: \(z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}\). A second activation (or softmax for classification) yields the final output \(\hat{y} = \psi(z^{(2)})\). The weights encode learned linear transformations, biases allow shifting, and activations introduce non‑linearity.

9. **Derive the back‑propagation update rule for a single weight in a network with a mean‑squared‑error loss. Show each step from loss gradient to weight adjustment.**
   - Loss: \(L = \frac{1}{2}(y - \hat{y})^2\).
   - Gradient w.r.t. output: \(\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y})\).
   - For a weight \(w\) in the last layer, \(\hat{y} = \phi(z)\) with \(z = w a + b\).
   - Chain rule: \(\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w} = -(y - \hat{y}) \cdot \phi'(z) \cdot a\).
   - Update (SGD): \(w \leftarrow w - \eta \frac{\partial L}{\partial w}\), where \(\eta\) is the learning rate.

10. **Compare and contrast the following optimizers: SGD with momentum, RMSProp, and Adam. Discuss how each algorithm adapts the learning rate and the impact on convergence speed and stability.**
    - **SGD with Momentum** accumulates a velocity \(v_t = \beta v_{t-1} + (1-\beta)\nabla L\) and updates \(w \leftarrow w - \eta v_t\). It smooths updates but uses a single global learning rate.
    - **RMSProp** maintains an exponential moving average of squared gradients \(s_t = \beta s_{t-1} + (1-\beta) (\nabla L)^2\) and scales the learning rate per parameter: \(w \leftarrow w - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla L\). This adapts to the geometry of the loss surface.
    - **Adam** combines momentum and RMSProp by keeping both first‑moment \(m_t\) and second‑moment \(v_t\) estimates, bias‑corrected, and updates: \(w \leftarrow w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}\). It often converges faster and is less sensitive to the initial learning rate.
    - In practice, Adam provides the most robust convergence on deep networks, while SGD with momentum can achieve better generalisation when carefully tuned.

11. **Discuss the trade‑offs involved in choosing an activation function for deep networks (e.g., ReLU vs. sigmoid vs. tanh). Include considerations of vanishing/exploding gradients and computational efficiency.**
    - **ReLU** is computationally cheap and mitigates vanishing gradients for positive inputs, but can suffer from “dead neurons” when inputs are negative. It does not bound activations, which can lead to exploding activations if not managed.
    - **Sigmoid** squashes outputs to (0, 1), useful for binary probabilities, but saturates for large magnitude inputs, causing vanishing gradients and slower training.
    - **Tanh** is zero‑centered, helping optimization, yet also saturates for large inputs, leading to vanishing gradients. It is more computationally expensive than ReLU.
    - Modern architectures typically favour ReLU‑family variants (Leaky ReLU, ELU) for depth, reserving sigmoid/tanh for output layers where bounded ranges are required.

12. **Design a short quiz (5 questions) that tests understanding of forward pass, backward pass, and optimizer selection. Provide the correct answer key separately.**
    - *Quiz*: 
      1. In a feed‑forward network, what operation follows the weighted sum of inputs? 
      2. Which algorithm computes gradients for all parameters using the chain rule? 
      3. Name an optimizer that adapts learning rates per parameter. 
      4. What problem does the ReLU activation help alleviate compared to sigmoid? 
      5. True or False: Momentum in SGD adds a fraction of the previous update to the current gradient step.
    - *Answer Key*: 
      1. Apply an activation function. 
      2. Back‑propagation. 
      3. RMSProp or Adam. 
      4. Vanishing gradients. 
      5. True.

*These paired questions and answers are ready for conversion to PDF for academic use.*

## Complex Exam Answers (Marks Distribution)

### 1‑Mark Answers
1. **MLP** – Multilayer Perceptron.
2. **Linear neuron output** – \(y = \mathbf{w}^T\mathbf{x} + b\).
3. **Epoch** – One full pass over the entire training dataset.
4. **Bias term** – Allows the activation to be shifted, enabling the model to fit data that does not pass through the origin.
5. **ReLU advantage** – Mitigates vanishing gradients by providing a gradient of 1 for positive inputs.
6. **Weight matrix shape** – \(64 \times 128\) (rows = output units, columns = input units).
7. **MSE loss** – \(L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2\).
8. **Adam** – Adapts learning rates per parameter using first‑ and second‑moment estimates.
9. **Back‑propagation computes** – The gradient of the loss with respect to each parameter.
10. **LaTeX gradient symbol** – \(\nabla\).

### 2‑Mark Answers
11. **Vanishing gradients** hinder learning because gradients become extremely small in deep layers, preventing effective weight updates.
12. **Gradient of MSE w.r.t. \(\hat{y}\)** – For a single sample with \(L = \frac{1}{2}(y - \hat{y})^2\), the gradient is \(\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y})\). For the mean over N samples with \(L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2\), the gradient is \(\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)\).
13. **Momentum** adds a fraction \(\beta\) of the previous update to the current gradient, accelerating convergence and reducing oscillations.
14. **Adam update** – \(m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t\), \(v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2\), bias‑corrected \(\hat{m}_t = m_t/(1-\beta_1^t)\), \(\hat{v}_t = v_t/(1-\beta_2^t)\), then \(\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}\).
15. **tanh vs sigmoid** – tanh outputs \((-1,1)\) and is zero‑centered; sigmoid outputs \((0,1)\) and is not zero‑centered, affecting gradient flow.
16. **Weight decay** adds \(\lambda \|\mathbf{w}\|_2^2\) to the loss, encouraging smaller weights.
17. **Softmax** – \(\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\).
18. **Learning‑rate decay** gradually reduces \(\eta\) during training, helping convergence to a minima.
19. **Batch vs mini‑batch** – Batch uses the whole dataset for each update; mini‑batch uses a subset, balancing noise and efficiency.
20. **Binary cross‑entropy** – \(L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]\).
21. **Over‑parameterization** – Having more parameters than training examples, which can still generalise due to implicit regularisation.
22. **Dropout** randomly zeroes activations during training, preventing co‑adaptation of neurons.
23. **Chain rule** – Computes gradients of composite functions by multiplying partial derivatives of each layer.
24. **Early stopping** monitors validation loss (or accuracy) and stops training when performance stops improving.
25. **L1 regularization** – \(\lambda \sum_i |w_i|\).

### 5‑Mark Answers
26. **Back‑prop for two‑layer MLP** – Derive \(\frac{\partial L}{\partial W^{(2)}} = (\hat{y}-y)\phi'(z^{(2)}) a^{(1)T}\) and \(\frac{\partial L}{\partial W^{(1)}} = ((W^{(2)T}(\hat{y}-y)\phi'(z^{(2)}))\odot \phi'(z^{(1)})) x^T\).
27. **ReLU zero gradient issue** – For \(x<0\) gradient is 0; Leaky ReLU uses \(\alpha x\) for negative inputs, preserving a small gradient.
28. **Cross‑entropy + softmax gradient** – \(\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i\).
29. **RMSProp update** – Maintain \(s_t = \beta s_{t-1} + (1-\beta)g_t^2\); update \(\theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t+\epsilon}} g_t\).
30. **Nesterov momentum** – Look‑ahead gradient: \(v_t = \beta v_{t-1} - \eta \nabla L(\theta - \beta v_{t-1})\), then \(\theta \leftarrow \theta + v_t\). Alternative formulation: \(\theta_{temp} = \theta + \beta v_{t-1}\), compute gradient at \(\theta_{temp}\), then \(v_t = \beta v_{t-1} - \eta \nabla L(\theta_{temp})\) and \(\theta \leftarrow \theta + v_t\).
31. **Batch norm effect** – Normalises activations to zero mean/unit variance, stabilising gradients and allowing higher learning rates.
32. **AdamW** – Decouples weight decay from Adam’s adaptive update: \(\theta \leftarrow \theta - \eta (\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda \theta)\).
33. **Conv output size** – With kernel 3, stride 1, padding 1, output height/width = \((32 + 2*1 - 3)/1 + 1 = 32\); output shape = \(32 \times 32 \times 16\).
34. **Hinge loss gradient** – For hinge loss \(L = \max(0, 1 - y(w^Tx))\) where \(y \in \{-1, +1\}\), the gradient is \(\frac{\partial L}{\partial w} = \begin{cases}0 & \text{if } y(w^Tx) \ge 1 \\ -yx & \text{if } y(w^Tx) < 1\end{cases}\).
35. **Gradient clipping** – Clip by norm: \(g \leftarrow g \cdot \frac{c}{\max(c, \|g\|)}\) where \(c\) is the clipping threshold.
36. **Cosine annealing schedule** – \(\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t\pi}{T}))\).
37. **Batch‑norm back‑prop** – Gradients w.r.t. \(\gamma, \beta\) and input derived using chain rule; see original BN paper for full derivation.
38. **Weight init** – He init: \(w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)\) or variance \(\text{Var}(w) = \frac{2}{n_{in}}\) for ReLU; Xavier: \(w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}+n_{out}}}\right)\) or variance \(\text{Var}(w) = \frac{2}{n_{in}+n_{out}}\) for tanh/sigmoid.
39. **Fisher Information Matrix** – \(F = \mathbb{E}_{x\sim p_{data}}[\nabla_\theta \log p(y|x;\theta) \nabla_\theta \log p(y|x;\theta)^T]\).
40. **Jacobian** – \(J_{ij}=\frac{\partial f_i(x)}{\partial x_j}\); compute via automatic differentiation or explicit formulas for each layer.

### 10‑Mark Answers
41. **MNIST three‑layer design** – Input 784 → hidden 128 (ReLU) → output 10 (softmax). Loss: cross‑entropy. Provide forward equations \(z^{(1)}=W^{(1)}x+b^{(1)}\), \(a^{(1)}=\text{ReLU}(z^{(1)})\), \(z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}\), \(\hat{y}=\text{softmax}(z^{(2)})\). Back‑prop gradients for each weight matrix derived using chain rule, as in standard textbook.
42. **SGD vs Adam vs AdaGrad vs RMSProp** – Discuss convergence speed (Adam fastest in practice), stability (SGD with momentum stable), generalisation (SGD often better), and mathematical update formulas.
43. **Second‑order update** – Newton’s method: \(\theta_{new}=\theta - H^{-1}\nabla L\) where \(H\) is Hessian; impractical due to \(O(n^2)\) cost and noise, leading to approximations like L‑BFGS.
44. **Transformer attention** – Scaled dot‑product: \(\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\). Gradient w.r.t. Q, K, V derived via chain rule; see Vaswani et al. 2017.
45. **VAE loss** – \(L = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))\). KL term for Gaussian: \(\frac{1}{2}\sum(\mu^2+\sigma^2-\log\sigma^2-1)\). Reparameterisation: \(z = \mu + \sigma \odot \epsilon\).
46. **Policy gradient** – \(\nabla J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) A(s,a)]\). Update: \(\theta \leftarrow \theta + \alpha \nabla J(\theta)\).
47. **GAN math** – Discriminator loss: \(L_D = -\mathbb{E}_{x\sim p_{data}}[\log D(x)] - \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]\). Generator loss: \(L_G = -\mathbb{E}_{z\sim p_z}[\log D(G(z))]\). Gradients derived via back‑prop through both networks.
48. **LRP** – Relevance propagation rule: \(R_j = \sum_k \frac{a_j w_{jk}}{\sum_i a_i w_{ik}+\epsilon}\, R_k\). Detailed derivation based on conservation of relevance.
49. **Optimal learning rate** – For quadratic loss \(L=\frac{1}{2}\theta^T H \theta\), optimal \(\eta = 2/(\lambda_{max}+\lambda_{min})\) where \(\lambda\) are eigenvalues of Hessian.
50. **Neural Tangent Kernel** – In the infinite‑width limit, NTK \(\Theta(x,x') = \nabla_\theta f(x;\theta)^T \nabla_\theta f(x';\theta)\). Derivation uses first‑order Taylor expansion of network output.

*End of complex exam answers.*

## Computational Challenge Answers (Parameter Counting & Calculations)

### FFNN Parameter Counting Answers (2-Mark)
51. **Total parameters**: 
    - Layer1: \((784 \times 256) + 256 = 200,960\)
    - Layer2: \((256 \times 128) + 128 = 32,896\)
    - Layer3: \((128 \times 10) + 10 = 1,290\)
    - **Total**: \(200,960 + 32,896 + 1,290 = 235,146\) parameters

52. **Find hidden layer size \(h\)**:
    - Parameters: \((512 \times h) + h + (h \times 5) + 5 = 100,000\)
    - Simplify: \(512h + h + 5h + 5 = 518h + 5 = 100,000\)
    - Solve: \(h = (100,000 - 5)/518 = 99,995/518 \approx 193\)
    - **Answer**: \(h = 193\)

53. **Compare architectures**:
    - (A) 784→512→10: \((784 \times 512) + 512 + (512 \times 10) + 10 = 401,408 + 512 + 5,120 + 10 = 407,050\)
    - (B) 784→256→256→10: \((784 \times 256) + 256 + (256 \times 256) + 256 + (256 \times 10) + 10 = 200,704 + 256 + 65,536 + 256 + 2,560 + 10 = 269,322\)
    - **Answer**: (B) has fewer parameters by \(407,050 - 269,322 = 137,728\) parameters

### CNN Parameter Counting Answers (5-Mark)
54. **Conv layer parameters**:
    - Filters: \(5 \times 5 \times 32 \times 64 = 51,200\)
    - Biases: \(64\)
    - **Total**: \(51,200 + 64 = 51,264\) parameters

55. **Full CNN architecture**:
    - Conv1: \((3 \times 3 \times 3 \times 16) + 16 = 432 + 16 = 448\)
    - Conv2: \((3 \times 3 \times 16 \times 32) + 32 = 4,608 + 32 = 4,640\)
    - FC1: After Conv layers, spatial size is \(8\times8\), so flattened = \(8 \times 8 \times 32 = 2,048\)
      - FC1 params: \((2,048 \times 128) + 128 = 262,144 + 128 = 262,272\)
    - FC2: \((128 \times 10) + 10 = 1,280 + 10 = 1,290\)
    - **Total**: \(448 + 4,640 + 262,272 + 1,290 = 268,650\) parameters

56. **Depthwise separable convolution**:
    - Standard conv: \((3 \times 3 \times 64 \times 64) + 64 = 36,864 + 64 = 36,928\)
    - Depthwise separable: 
      - Depthwise: \((3 \times 3 \times 64) + 64 = 576 + 64 = 640\)
      - Pointwise: \((1 \times 1 \times 64 \times 64) + 64 = 4,096 + 64 = 4,160\)
      - Total: \(640 + 4,160 = 4,800\)
    - **Reduction ratio**: \(36,928 / 4,800 \approx 7.69\) (about 87% parameter reduction)

### Dimension Calculation Answers (2-Mark)
57. **Dimension tracking**:
    - Input: \(224\times224\times3\)
    - After Conv(\(7\times7\), stride=2, padding=3, 64 filters): \(\lfloor(224+2\times3-7)/2\rfloor + 1 = \lfloor223/2\rfloor + 1 = 112\)
      - Output: \(112\times112\times64\)
    - After MaxPool(\(3\times3\), stride=2): \(\lfloor(112-3)/2\rfloor + 1 = 55\)
      - **Final output**: \(55\times55\times64\)

58. **Step-by-step dimensions**:
    - Input: \(28\times28\times1\)
    - After Conv(\(5\times5\), no padding, stride=1): \((28-5+1) = 24\), so \(24\times24\times32\)
    - After MaxPool(\(2\times2\), stride=2): \(\lfloor24/2\rfloor = 12\)
      - **Final output**: \(12\times12\times32\)

### Gradient Computation Answers (5-Mark)
59. **Numerical gradient calculation**:
    - Forward pass:
      - \(z^{(1)} = W^{(1)}x = \begin{bmatrix}0.5 & -0.3\\0.2 & 0.8\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}-0.1\\1.8\end{bmatrix}\)
      - \(a^{(1)} = \sigma(z^{(1)}) = \begin{bmatrix}0.475\\0.858\end{bmatrix}\) (sigmoid)
      - \(z^{(2)} = W^{(2)T}a^{(1)} = 0.4(0.475) - 0.6(0.858) = -0.325\)
      - \(\hat{y} = \sigma(z^{(2)}) = \sigma(-0.325) = 0.419\)
      - \(L = \frac{1}{2}(0 - 0.419)^2 = 0.088\)
    - Backward pass:
      - \(\delta^{(2)} = (0.419 - 0) \times \sigma'(-0.325) = 0.419 \times 0.230 = 0.096\)
      - \(\delta^{(1)} = (W^{(2)}\delta^{(2)}) \odot \sigma'(z^{(1)}) = \begin{bmatrix}0.038\\-0.058\end{bmatrix} \odot \begin{bmatrix}0.249\\0.122\end{bmatrix} = \begin{bmatrix}0.009\\-0.007\end{bmatrix}\)
      - \(\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)}x^T = \begin{bmatrix}0.009\\-0.007\end{bmatrix}\begin{bmatrix}1 & 2\end{bmatrix} = \begin{bmatrix}0.009 & 0.018\\-0.007 & -0.014\end{bmatrix}\)

60. **Conv gradient calculation**:
    - Forward: \(output = \sum W \odot X = 1(3) + 0(1) + (-1)(2) + 2(4) = 3 + 0 - 2 + 8 = 9\)
    - Loss: \(L = (9-5)^2 = 16\)
    - Gradient: \(\frac{\partial L}{\partial output} = 2(9-5) = 8\)
    - \(\frac{\partial L}{\partial W} = \frac{\partial L}{\partial output} \times X = 8 \times \begin{bmatrix}3 & 1\\2 & 4\end{bmatrix} = \begin{bmatrix}24 & 8\\16 & 32\end{bmatrix}\)

### Memory and FLOP Calculation Answers (5-Mark)
61. **Memory calculation**:
    - Input: \(32 \times 224 \times 224 \times 3 = 4,816,896\) values
    - After Conv1(64 filters, \(7\times7\)): assuming stride=1, padding=3: \(32 \times 224 \times 224 \times 64 = 102,760,448\)
    - After Conv2(128 filters, \(3\times3\)): assuming same spatial: \(32 \times 224 \times 224 \times 128 = 205,520,896\)
    - After FC(1000): \(32 \times 1000 = 32,000\)
    - Total values: \(\approx 313 \text{ million}\)
    - Memory: \(313,129,240 \times 4 \text{ bytes} = 1,252,516,960 \text{ bytes} \approx 1,195 \text{ MB}\)

62. **FLOPs calculation**:
    - For each output position: \(3 \times 3 \times 64\) multiplications + 64 bias adds = \(576 + 64 = 640\) ops
    - Output has \(56 \times 56 \times 128 = 401,408\) positions
    - **Total FLOPs**: \(401,408 \times 640 \approx 257 \text{ million FLOPs}\)
    - More precisely: \(56 \times 56 \times 128 \times (3 \times 3 \times 64 + 1) = 56 \times 56 \times 128 \times 577 = 231,616,512\) FLOPs

63. **FLOP comparison**:
    - (A) Standard \(3\times3\): \(56 \times 56 \times 128 \times (3 \times 3 \times 128) = 56 \times 56 \times 128 \times 1,152 = 462,422,016\) FLOPs
    - (B) Bottleneck:
      - \(1\times1\) (128→32): \(56 \times 56 \times 32 \times 128 = 12,845,056\)
      - \(3\times3\) (32→32): \(56 \times 56 \times 32 \times (3 \times 3 \times 32) = 56 \times 56 \times 32 \times 288 = 28,901,376\)
      - \(1\times1\) (32→128): \(56 \times 56 \times 128 \times 32 = 12,845,056\)
      - Total: \(54,591,488\) FLOPs
    - **Reduction**: (B) uses \(\approx 11.8\%\) of (A)'s FLOPs (8.5× reduction)

### Advanced Architecture Analysis Answers (10-Mark)
64. **Design CNN with 50K parameters**:
    - Example architecture:
      - Conv1: \(3\times3\times3\) → 16 filters: \(3\times3\times3\times16 + 16 = 448\)
      - Conv2: \(3\times3\times16\) → 32 filters: \(3\times3\times16\times32 + 32 = 4,640\)
      - MaxPool \(2\times2\)
      - Conv3: \(3\times3\times32\) → 64 filters: \(3\times3\times32\times64 + 64 = 18,496\)
      - MaxPool \(2\times2\), spatial now \(8\times8\)
      - Flatten: \(8\times8\times64 = 4,096\)
      - FC1: \(4,096\) → 128: \(4,096\times128 + 128 = 524,416\) (too many!)
    - **Revised**: Use Global Average Pooling instead
      - After Conv3: \(64\) channels → GAP → \(64\) values
      - FC: \(64\) → 10: \(64\times10 + 10 = 650\)
      - **Total**: \(448 + 4,640 + 18,496 + 650 = 24,234\) (need more capacity)
    - **Better design**: Add more conv layers before pooling, adjust filter counts to reach ~50K

65. **ResNet block analysis**:
    - (a) Parameters:
      - Conv1 (\(1\times1\), 64→16): \(1\times1\times64\times16 + 16 = 1,040\)
      - Conv2 (\(3\times3\), 16→16): \(3\times3\times16\times16 + 16 = 2,320\)
      - Conv3 (\(1\times1\), 16→64): \(1\times1\times16\times64 + 64 = 1,088\)
      - **Total**: \(4,448\) parameters
    - (b) FLOPs:
      - Conv1: \(56\times56\times16\times64 = 3,211,264\)
      - Conv2: \(56\times56\times16\times(3\times3\times16) = 56\times56\times16\times144 = 7,225,344\)
      - Conv3: \(56\times56\times64\times16 = 3,211,264\)
      - **Total**: \(\approx 13.6\) M FLOPs
    - (c) Memory (batch=16): \(16 \times (56\times56\times64 + 56\times56\times16 + 56\times56\times16 + 56\times56\times64) \times 4\text{ bytes} \approx 6.4\text{ MB}\)
    - (d) Gradient flow: Skip connection provides direct path, preventing vanishing gradients

66. **FFNN comprehensive analysis**:
    - (a) With biases: \((1024\times512 + 512) + (512\times256 + 256) + (256\times128 + 128) + (128\times10 + 10) = 524,800 + 131,328 + 32,896 + 1,290 = 690,314\)
    - Without biases: \(1024\times512 + 512\times256 + 256\times128 + 128\times10 = 688,408\)
    - (b) Memory: \(690,314 \times 4 = 2,761,256\text{ bytes} \approx 2.63\text{ MB}\)
    - (c) Largest layer (1024→512): \(524,800\) params; Smallest (128→10): \(1,290\) params; Ratio: \(\approx 407:1\)
    - (d) With dropout(0.5): Expected active neurons: input=1024, hidden=[256, 128, 64], output=10

### Backpropagation Numerical Answers (5-Mark)
67. **XOR forward-backward**:
    - For sample (0,0)→0:
      - Initialize weights randomly (example): \(W^{(1)} = \begin{bmatrix}0.5 & 0.3\\-0.2 & 0.4\end{bmatrix}\), \(W^{(2)} = \begin{bmatrix}0.6\\-0.5\end{bmatrix}\)
      - Forward: \(z^{(1)} = W^{(1)}[0,0]^T = [0,0]^T\), \(a^{(1)} = \sigma([0,0]) = [0.5, 0.5]^T\)
      - \(z^{(2)} = W^{(2)T}[0.5, 0.5]^T = 0.05\), \(\hat{y} = \sigma(0.05) = 0.5125\)
      - \(L = 0.5(0-0.5125)^2 = 0.131\)
      - Backward: \(\delta^{(2)} = (0.5125-0)\sigma'(0.05) = 0.127\)
      - Updates: \(W^{(2)} \leftarrow W^{(2)} - 0.1 \times \delta^{(2)}[0.5, 0.5]^T\)
    - (Full calculation requires complete initialization and multiple samples)

68. **Batch normalization gradients**:
    - Input: \(X = [1, 3, 5, 7]^T\)
    - Mean: \(\mu = (1+3+5+7)/4 = 4\)
    - Variance: \(\sigma^2 = ((1-4)^2+(3-4)^2+(5-4)^2+(7-4)^2)/4 = (9+1+1+9)/4 = 5\)
    - Normalized: \(\hat{X} = (X-4)/\sqrt{5+0.001} = (X-4)/2.236 = [-1.34, -0.45, 0.45, 1.34]^T\)
    - Output: \(Y = 2\hat{X} + 1 = [-1.68, 0.11, 1.89, 3.68]^T\)
    - Given \(\frac{\partial L}{\partial Y} = [1,1,1,1]^T\):
      - \(\frac{\partial L}{\partial \gamma} = \sum \frac{\partial L}{\partial Y_i}\hat{X}_i = 1(-1.34) + 1(-0.45) + 1(0.45) + 1(1.34) = 0\)
      - \(\frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial Y_i} = 4\)
      - \(\frac{\partial L}{\partial X}\) requires chain rule through normalization (complex calculation)

### Multi-Step Calculation Answers (10-Mark)
69. **Complete CIFAR-10 design**:
    - Architecture:
      - Conv1: \(3\times3\times3\)→32, ReLU, output \(32\times32\times32\)
      - Conv2: \(3\times3\times32\)→32, ReLU, MaxPool \(2\times2\), output \(16\times16\times32\)
      - Conv3: \(3\times3\times32\)→64, ReLU, output \(16\times16\times64\)
      - Conv4: \(3\times3\times64\)→64, ReLU, MaxPool \(2\times2\), output \(8\times8\times64\)
      - GAP: \(8\times8\times64\) → 64
      - FC: 64→10
    - Parameter count: \(896 + 9,248 + 18,496 + 36,928 + 650 = 66,218\) (adjust filters to reach ~1M if needed)
    - (Full answer requires receptive field calculation and memory analysis)

70. **VGG-16 first block**:
    - (a) Conv1 params: \(3\times3\times3\times64 + 64 = 1,792\)
    - Conv2 params: \(3\times3\times64\times64 + 64 = 36,928\)
    - (b) Total: \(38,720\) parameters
    - (c) Dimensions: \(224\times224\times3\) → \(224\times224\times64\) → \(224\times224\times64\) → \(112\times112\times64\)
    - (d) FLOPs: Conv1: \(224\times224\times64\times(3\times3\times3) \approx 86.7\)M; Conv2: \(224\times224\times64\times(3\times3\times64) \approx 1.85\)B
    - (e) Activation memory (batch=8): \(8\times(224\times224\times3 + 224\times224\times64 + 224\times224\times64)\times4 \approx 154\)MB
    - (f) Gradient memory: Same as activation memory for storing gradients

*End of computational challenge answers.*

