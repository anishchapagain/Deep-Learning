# Practice Questions – Neural Networks and Deep Learning

**Student Instructions**
- This document contains **short and long practice questions**.
- Every question includes **Preparation Hints**.
- Students are expected to prepare **both concise and long analytical answers**.
- Many answers are demonstrated or hinted at in **Lecture slides, Detailed course material supplied, Workshop notebooks, coding labs, and tutorial materials**. You are strongly advised to study them carefully.

**NOTE: Try to include pseudocode or code blocks, main keywords, math formulas during your answer**
**NOTE: Be concise to the question**

### Q1
A neural network with 3 layers has the following architecture:  
● Input layer (700 neurons with bias) + ReLU  
● Hidden layer (128 neurons with bias) + ReLU  
● Output layer (12 neurons with bias) + Sigmoid  
Calculate the total number of trainable parameters in this network.

**Preparation Hints**
- Recall how parameters are computed in fully connected layers.
- Bias contributes one parameter per neuron.
- Show step-by-step calculations clearly.
- Practice similar examples from workshop problem sets.



### Q2
Calculate feature map size for the provided details:  
An input image has size 72 × 72 × 3. A convolution layer uses:  
● Kernel size = 4 × 4  
● Stride = 1  
● Padding = 2  
● Number of filters = 32  

a. Calculate the output feature map size.  
b. Explain why padding = 2 is chosen here.

**Preparation Hints**
- Use the convolution output size formula.
- Understand the concept of “same padding”.
- Explain padding using diagrams from CNN lectures.



### Q3
Design a PyTorch ANN for binary classification on tabular data (20 features).  
Requirements:  
● Two hidden layers  
● Appropriate activation functions  
● Output suitable for binary classification

**Preparation Hints**
- Review `nn.Module` and `forward()` structure.
- Understand why ReLU is used in hidden layers.
- Know why Sigmoid pairs with Binary Cross-Entropy.
- Revisit workshop coding templates.



### Q4
Do you agree deep learning models generally outperform traditional machine learning approaches on large-scale unstructured data (such as images or text)? Justify your answer with at least two specific reasons and few examples.

**Preparation Hints**
- Focus on automatic feature extraction.
- Compare deep learning with manual feature engineering.
- Use real-world examples discussed in lectures (vision, NLP).



### Q5
During the forward pass of a neural network, what is the sequence of operations that occur at each neuron? Describe the mathematical process.

**Preparation Hints**
- Weighted sum, bias addition, activation.
- Write equations such as z = Wx + b.
- Explain how this repeats layer by layer.



### Q6
Explain Backpropagation Through Time (BPTT) in RNNs. Why is it sensitive to vanishing gradients? Provide a solution.

**Preparation Hints**
- Understand unrolling of RNNs through time.
- Apply chain rule reasoning.
- Mention LSTM, GRU, and gradient clipping.



### Q7
Explain Dropout regularization and critically analyze:  
● How it operates during training  
● Why it is disabled during evaluation  
● How Dropout differs conceptually from weight-based regularization methods

**Preparation Hints**
- Dropout randomly deactivates neurons.
- Explain `model.train()` vs `model.eval()`.
- Compare Dropout with L1/L2 regularization.



### Q8
For the following tasks:  
● Image classification  
● Speech-to-text translation  
● Stock price prediction  
● Sentiment analysis on text  
● Credit risk prediction  

Identify the most suitable model or architecture (ML, ANN, CNN, RNN, Encoder–Decoder) for each task and justify your choices briefly.

**Preparation Hints**
- Match data structure to model architecture.
- Use examples practiced during workshops.
- Justify each choice with one clear technical reason.



### Q9
Softmax activation is commonly used in multi-class classification problems.  
(a) Write the mathematical formula for Softmax and explain its properties.  
(b) Explain why Softmax is preferred over using multiple Sigmoid outputs for multi-class classification.  
(c) Calculate the Softmax output for the logit vector z = [2.0, 1.0, 0.1] and interpret the results.

**Preparation Hints**
- Practice manual Softmax calculation.
- Understand probability normalization.
- Link Softmax with categorical cross-entropy loss.



### Q10
The Self-Attention mechanism is the core innovation of the Transformer architecture.  
(a) Describe the roles of the Query (Q), Key (K), and Value (V) vectors in the calculation of attention.  
(b) Discuss two reasons why Self-Attention is superior to Recurrent Neural Networks (RNNs) for processing long sequences.

**Preparation Hints**
- Review scaled dot-product attention.
- Focus on parallelism and long-range dependency capture.



### Q11
Compare Mean Squared Error (MSE), Binary Cross-Entropy, and Categorical Cross-Entropy loss functions. For each, provide the mathematical formula, typical use cases, and explain when to use which loss function.

**Preparation Hints**
- Memorize formulas.
- Understand loss–activation pairing.
- Use simple examples to explain.



### Q12
Explain the architecture and working of a Vanilla Autoencoder. Discuss at least three different types of Autoencoders and their applications. Also, explain how the bottleneck layer enforces a compressed representation and why this is useful.

**Preparation Hints**
- Revise encoder–latent–decoder flow.
- Understand denoising, sparse, and convolutional autoencoders.
- Link bottleneck to feature learning.



### Q13
Compare VAE and GAN for generative modeling. Include architecture, training process, and a few pros and cons of each.

**Preparation Hints**
- Likelihood-based vs adversarial training.
- Stability vs output sharpness.
- Revise KL divergence and discriminator loss.



### Q14
Compare and contrast Inductive Transfer Learning, Transductive Transfer Learning, and Unsupervised Transfer Learning.

**Preparation Hints**
- Focus on data availability.
- Explain domain similarity vs task similarity.
- Prepare one real-world example for each.



### Q15
Explain gradient descent, its role in deep learning, and compare its variants.

**Preparation Hints**
- Batch, Stochastic, and Mini-batch GD.
- Advantages and limitations of each.
- Use update rule equations conceptually.



### Q16
Explain the architecture of LSTM with all gates and equations. Compare LSTM and GRU, and explain when you would choose one over the other.

**Preparation Hints**
- Memorize gate equations.
- Compare parameter count and efficiency.
- Use sequence-length-based justification.



### Q17
Explain the Markov Property and why it is essential for Q-learning.

**Preparation Hints**
- Understand state sufficiency.
- Link to Bellman equation.



### Q18
Define the Exploration–Exploitation dilemma and explain epsilon-greedy policy.

**Preparation Hints**
- Trade-off between learning and reward.
- Explain epsilon with practical example.



### Q19
Describe the complete training life cycle of a neural network for a supervised learning problem.

**Preparation Hints**
- Data → preprocessing → model → training → evaluation → deployment.
- Emphasize monitoring and feedback loops.



### Q20
A neural network shows high training accuracy but poor production performance. Analyze causes and propose corrective actions.

**Preparation Hints**
- Overfitting, data leakage, data drift.
- Propose solutions at data, model, and deployment stages.



### Q21
Explain why ReLU is preferred over Sigmoid in deep networks. Also discuss the dying ReLU problem.

**Preparation Hints**
- Compare gradient behavior.
- Link to vanishing gradients.
- Mention Leaky ReLU and PReLU.



### Q22
What is Batch Normalization? Explain how it stabilizes and accelerates training.

**Preparation Hints**
- Normalization of layer inputs.
- Reduction of internal covariate shift.
- Training vs inference behavior.



### Q23
Explain mathematically why vanishing gradients occur in deep neural networks.

**Preparation Hints**
- Chain rule multiplication.
- Role of activation derivatives.
- Use simple numeric illustration.



### Q24
Why are CNNs more suitable for image data compared to fully connected networks?

**Preparation Hints**
- Local connectivity.
- Weight sharing.
- Translation invariance.



### Q25
Explain why positional encoding is required in Transformers.

**Preparation Hints**
- Transformers lack recurrence.
- Order information must be injected.
- Sinusoidal vs learned encodings.



### Q26
Compare Encoder-only, Decoder-only, and Encoder–Decoder Transformer architectures.

**Preparation Hints**
- BERT vs GPT vs Seq-to-Seq.
- Use cases for each.



### Q27
Explain how attention mechanisms differ fundamentally from recurrence.

**Preparation Hints**
- Direct connections vs sequential dependency.
- Computational complexity discussion.



### Q28
Explain the role of bottleneck layers in representation learning beyond autoencoders.

**Preparation Hints**
- Dimensionality reduction.
- Feature compression.
- Use in transfer learning and NLP.



### Q29
Explain experience replay and target networks in Deep Q-Networks.

**Preparation Hints**
- Correlation reduction.
- Stabilizing Q-learning.
- Link to reinforcement learning lectures.



### Q30
Explain how learning rate scheduling affects convergence and generalization.

**Preparation Hints**
- Step decay, exponential decay, cosine annealing.
- Early vs late training behavior.



### Q31
Explain the difference between classification and regression problems in neural networks.

**Preparation Hints**
- Output type.
- Loss functions.
- Activation functions.



### Q32
Discuss ethical risks associated with generative models and propose mitigation practices.

**Preparation Hints**
- Bias, privacy, deepfakes.
- Responsible AI practices from lectures.



### Q33
What factors influence the choice of batch size during training?

**Preparation Hints**
- Memory constraints.
- Gradient stability.
- Generalization effects.



### Q34
Explain common mistakes in PyTorch training loops and how to avoid them.

**Preparation Hints**
- Forgetting zero_grad().
- Incorrect train/eval mode.
- Shape mismatches.



### Q35
Explain why loss may decrease while accuracy does not improve significantly.

**Preparation Hints**
- Class imbalance.
- Decision threshold issues.
- Loss vs metric mismatch.



### Q36
Explain how over-parameterization can both help and hurt deep learning models.

**Preparation Hints**
- Expressive power vs overfitting.
- Role of regularization.



### Q37
Explain the role of KL divergence in Variational Autoencoders.

**Preparation Hints**
- Regularizing latent space.
- Approximate posterior vs prior.



### Q38
Explain how transfer learning reduces training time and data requirements.

**Preparation Hints**
- Feature reuse.
- Freezing layers.
- Domain similarity.



### Q39
Explain why gradient clipping is important in training RNNs.

**Preparation Hints**
- Exploding gradients.
- Stabilizing updates.



### Q40
Explain the difference between model capacity and model generalization.

**Preparation Hints**
- Bias–variance tradeoff.
- Practical examples from lectures.

### Q41
How a complete model trainign takes place, say using PyTorch?
- Pick a problem and try to show all major steps as code block or pseudocode.

### Q42
Explain few activation functions and optimizers
- Include their math formulas
- Try to explain their key characteristics with pros and cons

## Final Advice
- Prepare for **both mathematical explanations and conceptual reasoning**.
- Practice **writing long/short answers** clearly.
- Revisit **lecture slides, workshops, and coding labs** for hidden clues and depth.
