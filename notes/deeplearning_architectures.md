# Deep Learning Architectures

Overview of various deep learning architectures.

---

## 1. Multi-layer Perceptrons (MLP)

### Description

A Multi-layer Perceptron (MLP) is a fundamental class of feedforward artificial neural network (ANN). It consists of an input layer, one or more hidden layers, and an output layer. Each layer is composed of nodes (neurons), and every node in a layer is connected to every node in the subsequent layer, making it a "fully connected" network. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. This allows MLPs to learn complex, non-linear relationships in data. MLPs are trained using supervised learning, with backpropagation being the most common algorithm for updating the network's weights.

![MLP Architecture](https://www.deeplearning.ai/wp-content/uploads/2023/04/multilayer-perceptron-architecture.png)

### Pros and Cons

**Pros:**
- **Universal Approximators:** MLPs can approximate any continuous function, making them powerful for a wide range of problems.
- **Flexibility:** They can be used for both classification and regression tasks.
- **Simple to understand and implement:** The architecture is straightforward compared to more complex models.

**Cons:**
- **Scalability:** The number of parameters can grow very large with many layers and neurons, leading to high computational cost and memory usage.
- **Lack of spatial/temporal awareness:** MLPs do not handle sequential or grid-like data (like images) efficiently, as they do not consider the input's structure.
- **Prone to overfitting:** With a large number of parameters, MLPs can easily overfit the training data if not regularized properly.

### Mathematical Formulas

The output of a single neuron is given by:

$$ y = f(\sum_{i=1}^{n} w_i x_i + b) $$

Where:
- $y$ is the output of the neuron.
- $f$ is the activation function.
- $w_i$ are the weights of the connections from the previous layer.
- $x_i$ are the inputs from the previous layer.
- $b$ is the bias.

For a simple one-hidden-layer MLP, the output is:

$$ \text{Output} = f_2(W_2 \cdot f_1(W_1 \cdot X + b_1) + b_2) $$

Where:
- $X$ is the input vector.
- $W_1$ and $W_2$ are the weight matrices for the first and second layers, respectively.
- $b_1$ and $b_2$ are the bias vectors for the first and second layers, respectively.
- $f_1$ and $f_2$ are the activation functions for the hidden and output layers, respectively.

### PyTorch Code Example

The following code defines and trains a simple MLP for a classification task. Note that we use dummy data for demonstration purposes.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = 784  # e.g., 28x28 images flattened
hidden_size = 500
output_size = 10  # e.g., 10 classes for MNIST
learning_rate = 0.001
batch_size = 100
num_epochs = 5

# Dummy data loader for demonstration
# In a real scenario, you would use a dataset like MNIST
dummy_data = torch.randn(batch_size * 10, input_size)
dummy_labels = torch.randint(0, output_size, (batch_size * 10,))
train_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model, loss, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

```

### Useful Web Links

- [PyTorch Tutorial on Building a Neural Network](https://pytorch.org/tutorials/beginner/basics/build_model_tutorial.html)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Stanford CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)

---

## 2. Convolutional Neural Networks (CNN)

### Description

Convolutional Neural Networks (CNNs or ConvNets) are a class of deep neural networks specifically designed for processing grid-like data, such as images. They are inspired by the organization of the animal visual cortex. CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features from the input. A typical CNN architecture consists of one or more convolutional layers, followed by pooling layers, and then one or more fully connected layers for classification or regression.

![CNN Architecture](https://miro.medium.com/max/1400/1*1CI_2s_tP88kSO5e_g3A_A.jpeg)

### Pros and Cons

**Pros:**
- **Parameter Sharing:** CNNs use shared weights in convolutional layers, which significantly reduces the number of parameters and makes them more efficient than fully connected networks for image data.
- **Translation Invariance:** Due to the nature of convolution and pooling, CNNs are inherently translation invariant, meaning they can recognize an object regardless of its position in the image.
- **Hierarchical Feature Learning:** CNNs learn features in a hierarchical manner, from simple edges and textures in the initial layers to more complex objects in deeper layers.

**Cons:**
- **Computational Cost:** Deep CNNs can be computationally expensive to train, requiring significant hardware resources (GPUs).
- **Large Datasets Required:** CNNs typically require large amounts of labeled data to achieve high performance and avoid overfitting.
- **Less effective for non-grid data:** They are not naturally suited for tasks involving non-grid data like text or time series.

### Mathematical Formulas

The core operation in a CNN is the convolution. For a 2D input image $I$ and a 2D kernel $K$, the convolution is defined as:

$$ S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(m, n) K(i - m, j - n) $$

In practice, neural network libraries implement this as a cross-correlation:

$$ S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i + m, j + n) K(m, n) $$

After convolution, an activation function is applied. A pooling operation is often used to reduce the spatial dimensions. The most common is max pooling:

$$ P(i, j) = \max_{m, n} A(i \cdot s + m, j \cdot s + n) $$

Where $A$ is the input to the pooling layer, and $s$ is the stride.

### PyTorch Code Example

The following code defines a simple CNN for image classification.

```python
import torch
import torch.nn as nn

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 5

# Dummy data (assuming MNIST-like 28x28 images)
dummy_images = torch.randn(batch_size, 1, 28, 28)

# Model
model = CNN(num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Dummy labels
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    outputs = model(dummy_images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

```

### Useful Web Links

- [PyTorch Tutorial on Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
- [A guide to convolution arithmetic for deep learning](https://github.com/vdumoulin/conv_arithmetic)

---

## 3. Recurrent Neural Networks (RNN)

### Description

Recurrent Neural Networks (RNNs) are a type of neural network designed for sequential data. Unlike feedforward networks, RNNs have connections that form directed cycles, allowing them to maintain an internal state or "memory". This memory enables them to capture temporal dependencies and process sequences of inputs, making them suitable for tasks like natural language processing, speech recognition, and time series analysis.

![RNN Architecture](https://miro.medium.com/max/1400/1*yBXVVb-lI3b-Q9H4dc_l-A.gif)

### Pros and Cons

**Pros:**
- **Handles Sequential Data:** RNNs are specifically designed to model sequences and capture temporal dependencies.
- **Parameter Sharing:** They share parameters across time steps, making them efficient for variable-length sequences.

**Cons:**
- **Vanishing/Exploding Gradients:** Standard RNNs struggle to learn long-term dependencies due to the vanishing or exploding gradient problem.
- **Sequential Processing:** The recurrent nature prevents parallelization across time steps, making them slower to train than some other architectures.

### Mathematical Formulas

The state of an RNN at time $t$ is given by:

$$ h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h) $$

The output at time $t$ is:

$$ y_t = W_{hy} h_t + b_y $$

Where:
- $h_t$ is the hidden state at time $t$.
- $x_t$ is the input at time $t$.
- $h_{t-1}$ is the hidden state at the previous time $t-1$.
- $W_{hh}$, $W_{xh}$, $W_{hy}$ are weight matrices.
- $b_h$, $b_y$ are bias vectors.
- $f$ is an activation function (e.g., tanh or ReLU).

### PyTorch Code Example

```python
import torch
import torch.nn as nn

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10
hidden_size = 32
num_layers = 2
output_size = 1
sequence_length = 20
batch_size = 64
num_epochs = 2
learning_rate = 0.01

# Dummy data
dummy_sequence = torch.randn(batch_size, sequence_length, input_size)
dummy_labels = torch.randn(batch_size, output_size)

# Model
model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(dummy_sequence)
    loss = criterion(outputs, dummy_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### Useful Web Links

- [PyTorch Tutorial on Sequence-to-Sequence Modeling](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

## 4. Long Short-Term Memory (LSTM)

### Description

Long Short-Term Memory (LSTM) networks are a special kind of RNN, designed to address the vanishing gradient problem and learn long-term dependencies. LSTMs have a more complex cell structure than simple RNNs, which includes a cell state and three "gates" (input, forget, and output gates). These gates regulate the flow of information, allowing the network to selectively remember or forget information over long sequences.

![LSTM Architecture](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

### Pros and Cons

**Pros:**
- **Long-Term Dependencies:** LSTMs are very effective at capturing long-term dependencies in sequential data.
- **Avoids Vanishing Gradients:** The gating mechanism helps to mitigate the vanishing gradient problem.

**Cons:**
- **Complexity:** LSTMs are more complex than simple RNNs and GRUs, with more parameters to train.
- **Computational Cost:** The added complexity leads to higher computational cost during training.

### Mathematical Formulas

An LSTM has a cell state and three gates: an input gate, a forget gate, and an output gate.

- **Forget Gate:** Decides what information to throw away from the cell state.
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

- **Input Gate:** Decides which values we'll update.
$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
$$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

- **Cell State Update:**
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$

- **Output Gate:** Decides what to output.
$$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
$$ h_t = o_t * \tanh(C_t) $$

Where:
- $x_t$ is the input vector.
- $h_t$ is the hidden state vector.
- $C_t$ is the cell state vector.
- $W$ and $b$ are weight matrices and bias vectors.
- $\sigma$ is the sigmoid function.

### PyTorch Code Example

```python
import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10
hidden_size = 32
num_layers = 2
output_size = 1
sequence_length = 20
batch_size = 64
num_epochs = 2
learning_rate = 0.01

# Dummy data
dummy_sequence = torch.randn(batch_size, sequence_length, input_size)
dummy_labels = torch.randn(batch_size, output_size)

# Model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(dummy_sequence)
    loss = criterion(outputs, dummy_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### Useful Web Links

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Illustrated Guide to LSTMs and GRUs](https://www.youtube.com/watch?v=8HyCNIVRbSU)

---

## 5. Gated Recurrent Unit (GRU)

### Description

The Gated Recurrent Unit (GRU) is another type of recurrent neural network with a gating mechanism, similar to LSTMs. Introduced in 2014, GRUs have a simpler architecture than LSTMs, with two gates (a reset gate and an update gate) and no separate cell state. This makes them computationally more efficient while often achieving comparable performance to LSTMs.

![GRU Architecture](https://miro.medium.com/max/1400/1*A2-TD6n_v_H3t30p_s_pEg.png)

### Pros and Cons

**Pros:**
- **Efficiency:** GRUs have fewer parameters than LSTMs, making them faster to train.
- **Good Performance:** They often perform on par with LSTMs on many tasks.

**Cons:**
- **Less Expressive:** On some tasks, particularly those requiring very long-term dependencies, LSTMs may outperform GRUs due to their more complex gating mechanism and separate cell state.

### Mathematical Formulas

A GRU has two gates: a reset gate and an update gate.

- **Reset Gate:** Decides how to combine the new input with the previous memory.
$$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) $$

- **Update Gate:** Decides how much of the previous memory to keep around.
$$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) $$

- **Candidate Hidden State:**
$$ \tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1}, x_t] + b_h) $$

- **Final Hidden State:**
$$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $$

Where:
- $x_t$ is the input vector.
- $h_t$ is the hidden state vector.
- $W$ and $b$ are weight matrices and bias vectors.
- $\sigma$ is the sigmoid function.

### PyTorch Code Example

```python
import torch
import torch.nn as nn

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10
hidden_size = 32
num_layers = 2
output_size = 1
sequence_length = 20
batch_size = 64
num_epochs = 2
learning_rate = 0.01

# Dummy data
dummy_sequence = torch.randn(batch_size, sequence_length, input_size)
dummy_labels = torch.randn(batch_size, output_size)

# Model
model = GRUModel(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(dummy_sequence)
    loss = criterion(outputs, dummy_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### Useful Web Links

- [Illustrated Guide to LSTMs and GRUs](https://www.youtube.com/watch?v=8HyCNIVRbSU)
- [PyTorch GRU documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Animated RNN, LSTM and GRU](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)

---

## 6. Transformers

### Description

The Transformer is a neural network architecture that relies on self-attention mechanisms instead of recurrence (RNNs) or convolutions (CNNs). It was introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017). Transformers have become the state-of-the-art for many NLP tasks.

The key innovation is the self-attention mechanism, which allows the model to weigh the importance of different words in the input sequence. This enables parallel processing of the input sequence, making it much faster than sequential RNNs.

![Transformer Architecture](http://jalammar.github.io/images/t/transformer_architecture.png)

### Pros and Cons

**Pros:**
- **Parallelization:** Transformers can process entire sequences in parallel, leading to significantly faster training times compared to RNNs.
- **Long-Range Dependencies:** The self-attention mechanism allows Transformers to capture long-range dependencies more effectively than RNNs.
- **State-of-the-art Performance:** Transformers have achieved state-of-the-art results on many NLP benchmarks.

**Cons:**
- **High Computational Cost:** The self-attention mechanism has a quadratic complexity with respect to the sequence length, making it computationally expensive for very long sequences.
- **Large Data Requirement:** Transformers are data-hungry and require large datasets to train effectively.
- **Lack of Inductive Bias:** Unlike RNNs (sequential bias) or CNNs (spatial bias), Transformers have a weaker inductive bias, which can be a disadvantage on smaller datasets.

### Mathematical Formulas

The core of the Transformer is the scaled dot-product attention:

$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

Where:
- $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices derived from the input embeddings.
- $d_k$ is the dimension of the key vectors.

Transformers use multi-head attention, which runs the attention mechanism multiple times in parallel and concatenates the results.

### PyTorch Code Example

The following code is a simplified implementation of a Transformer model, based on the PyTorch tutorials.

```python
import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Define a simple Transformer model
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

# Note: This is a simplified example. A full implementation would require more components.

```

### Useful Web Links

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Tutorial on Transformers](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)

---

## 7. Autoencoders

### Description

An autoencoder is a type of artificial neural network used for unsupervised learning, primarily for dimensionality reduction and feature learning. It consists of two main parts: an encoder and a decoder. The encoder compresses the input into a lower-dimensional latent representation (the "bottleneck"), and the decoder reconstructs the input from this latent representation. The network is trained to minimize the reconstruction error, forcing it to learn the most important features of the data.

![Autoencoder Architecture](https://miro.medium.com/max/1400/1*44eDE2862A52243_cfv2fA.png)

### Pros and Cons

**Pros:**
- **Unsupervised Learning:** Autoencoders can be trained on unlabeled data.
- **Dimensionality Reduction:** They can learn compressed representations of data, which can be useful for visualization or as input to other models.
- **Feature Learning:** They can learn meaningful features from the data.

**Cons:**
- **Reconstruction Loss:** The model is optimized for reconstruction, which may not always result in the most useful latent representations for other tasks.
- **Information Loss:** The compression to a lower-dimensional space can lead to information loss.

### Mathematical Formulas

The encoder function $h = f(x)$ maps the input $x$ to a hidden representation $h$. The decoder function $r = g(h)$ reconstructs the input from the hidden representation.

The goal is to minimize the reconstruction error:

$$ L(x, g(f(x))) $$

Where $L$ is a loss function, for example, mean squared error (MSE).

### PyTorch Code Example

```python
import torch
import torch.nn as nn

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, encoding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, input_dim), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
input_dim = 784
encoding_dim = 32
num_epochs = 5
learning_rate = 1e-3

# Dummy data
dummy_data = torch.randn(100, input_dim)

# Model
model = Autoencoder(input_dim, encoding_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_data)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

```

### Useful Web Links

- [Building Autoencoders in PyTorch](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/08-deep-autoencoders.html)
- [Deep Autoencoders Tutorial](https://www.deeplearning.ai/materials/summaries/unsupervised-learning-generative-ai-and-deep-learning-specialization-course-1-introduction-to-deep-learning-and-neural-networks-with-keras/)
- [Auto-Encoders in PyTorch](https://www.geeksforgeeks.org/auto-encoders-in-pytorch/)

---

## 8. Generative Adversarial Networks (GANs)

### Description

Generative Adversarial Networks (GANs) are a class of generative models that consist of two neural networks: a generator and a discriminator, which are trained simultaneously in an adversarial manner. The generator's goal is to create realistic data (e.g., images), while the discriminator's goal is to distinguish between real and fake data. Through this adversarial process, the generator learns to produce increasingly realistic data.

![GAN Architecture](https://miro.medium.com/max/1400/1*X_G1hI-eSA521t_i3hX-4A.png)

### Pros and Cons

**Pros:**
- **Realistic Data Generation:** GANs can generate high-quality, realistic data.
- **Unsupervised/Semi-supervised Learning:** They can be trained with unlabeled data.
- **Versatile:** GANs have been applied to a wide range of tasks, including image generation, style transfer, and data augmentation.

**Cons:**
- **Training Instability:** GANs are notoriously difficult to train, often suffering from problems like mode collapse and non-convergence.
- **Difficult to Evaluate:** Evaluating the performance of a GAN can be challenging, as there is no single objective metric.

### Mathematical Formulas

The GAN loss function is a minimax game:

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

Where:
- $G$ is the generator.
- $D$ is the discriminator.
- $p_{data}(x)$ is the distribution of real data.
- $p_z(z)$ is the distribution of the noise input to the generator.
- $G(z)$ is the output of the generator.
- $D(x)$ is the probability that $x$ is real data.

### PyTorch Code Example

The following code provides a more complete example of a GAN, including the training loop for both the generator and discriminator.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
latent_dim = 100
img_dim = 784
learning_rate = 0.0002
batch_size = 64
num_epochs = 50

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# Initialize generator and discriminator
generator = Generator(latent_dim, img_dim)
discriminator = Discriminator(img_dim)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Dummy data loader
dummy_data = torch.randn(batch_size * 10, img_dim)
dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
train_loader = torch.utils.data.DataLoader(dataset=dummy_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs,) in enumerate(train_loader):

        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1, requires_grad=False)
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False)

        # Configure input
        real_imgs = imgs.float()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.size(0), latent_dim)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(
        f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
    )

```

### Useful Web Links

- [PyTorch Tutorial on DCGANs](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GANs in PyTorch](https://www.geeksforgeeks.org/generative-adversarial-network-gan-using-pytorch/)
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

---
