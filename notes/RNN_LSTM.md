# Long Short-Term Memory (LSTM)

## Introduction

Long Short-Term Memory (LSTM) is a special kind of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies. RNNs are a powerful tool for processing sequential data, but they suffer from the vanishing gradient problem, which makes it difficult for them to learn long-term dependencies. LSTMs were designed to overcome this problem.

The key idea behind LSTMs is the cell state, which is a horizontal line running through the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

## Why LSTMs?

Standard RNNs suffer from the vanishing and exploding gradient problems. When training an RNN, the gradients can become very small or very large, which makes it difficult to update the weights of the network. This is especially true for long sequences, where the gradients have to be propagated back through many time steps.

LSTMs were designed to address this problem by introducing the cell state and gates. The cell state allows information to flow through the network without being modified, while the gates control the flow of information into and out of the cell state. This allows LSTMs to learn long-term dependencies that standard RNNs cannot.

## Architecture and Formulas

An LSTM has a chain-like structure, like an RNN, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![LSTM Architecture](https://www.researchgate.net/profile/Md-Abdullah-Al-Mamun/publication/342222662/figure/fig1/AS:903246422351872@1592354742883/The-structure-of-the-long-short-term-memory-LSTM-neural-network-The-LSTM-cell.png)

The LSTM cell has three gates:

*   **Forget Gate:** Decides what information to throw away from the cell state.
*   **Input Gate:** Decides what new information to store in the cell state.
*   **Output Gate:** Decides what to output from the cell state.

### Forget Gate

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at `h_{t-1}` and `x_t`, and outputs a number between 0 and 1 for each number in the cell state `C_{t-1}`. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”

![Forget Gate](https://www.researchgate.net/profile/Md-Abdullah-Al-Mamun/publication/342222662/figure/fig3/AS:903246422360064@1592354742945/The-forget-gate-of-an-LSTM-cell.png)

The formula for the forget gate is:

`f_t = σ(W_f * [h_{t-1}, x_t] + b_f)`

### Input Gate

The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, `C̃_t`, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

![Input Gate](https://www.researchgate.net/profile/Md-Abdullah-Al-Mamun/publication/342222662/figure/fig4/AS:903246422364160@1592354742966/The-input-gate-of-an-LSTM-cell.png)

The formulas for the input gate are:

`i_t = σ(W_i * [h_{t-1}, x_t] + b_i)`

`C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)`

Now, we update the old cell state, `C_{t-1}`, into the new cell state `C_t`. The previous steps already decided what to do, we just need to actually do it.

`C_t = f_t * C_{t-1} + i_t * C̃_t`

### Output Gate

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between -1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

![Output Gate](https://www.researchgate.net/profile/Md-Abdullah-Al-Mamun/publication/342222662/figure/fig5/AS:903246426558464@1592354743008/The-output-gate-of-an-LSTM-cell.png)

The formulas for the output gate are:

`o_t = σ(W_o * [h_{t-1}, x_t] + b_o)`

`h_t = o_t * tanh(C_t)`

## Long and Short-Term Memory in LSTMs

The "long short-term memory" in an LSTM can be broken down into two components:

*   **Long-Term Memory (Cell State):** The **cell state (`C_t`)** is the key to the LSTM's ability to handle long-term dependencies. It acts as a "conveyor belt" of information, running through the entire sequence. The information in the cell state can be preserved for long periods, allowing the LSTM to "remember" things from many time steps ago. The gates of the LSTM can add or remove information from the cell state, but they do so with care, ensuring that important long-term information is not lost.

*   **Short-Term Memory (Hidden State):** The **hidden state (`h_t`)** can be thought of as the "short-term" or "working" memory of the LSTM. It is the output of the LSTM at each time step and is used to make predictions. The hidden state is a filtered version of the cell state, containing only the information that is relevant for the current time step.

The interplay between the cell state and the hidden state, regulated by the gates, is what gives LSTMs their power. The cell state holds the long-term context, while the hidden state provides the short-term information needed for the task at hand.

## Real-World Flow: Sentence Prediction with Numerical Example

Let's walk through a simplified numerical example of sentence prediction to understand the flow of an LSTM. Consider the sentence: "I love to play..." We want to predict the next word.

First, we need to convert the words into a numerical representation. Let's assume we have a simple embedding of dimension 2.

*   `E("I")` = `[0.1, 0.9]`
*   `E("love")` = `[0.8, 0.2]`
*   `E("to")` = `[0.3, 0.7]`
*   `E("play")` = `[0.6, 0.4]`

Let's also assume our LSTM has a hidden dimension of 2. We'll need placeholder weights and biases for our gates. For simplicity, let's assume all weight matrices are `1`s and all bias vectors are `0`s.

**Time Step 1: Input "I"**

*   `x_1 = [0.1, 0.9]`
*   `h_0 = [0, 0]`
*   `C_0 = [0, 0]`

Now, let's calculate the values for the gates:

*   **Forget Gate:** `f_1 = σ(W_f * [h_0, x_1] + b_f) = σ([[1, 1], [1, 1]] * [0, 0, 0.1, 0.9] + [0, 0]) = σ([1, 1]) = [0.73, 0.73]`
*   **Input Gate:** `i_1 = σ(W_i * [h_0, x_1] + b_i) = σ([[1, 1], [1, 1]] * [0, 0, 0.1, 0.9] + [0, 0]) = σ([1, 1]) = [0.73, 0.73]`
*   **Candidate Cell State:** `C̃_1 = tanh(W_C * [h_0, x_1] + b_C) = tanh([[1, 1], [1, 1]] * [0, 0, 0.1, 0.9] + [0, 0]) = tanh([1, 1]) = [0.76, 0.76]`
*   **Cell State:** `C_1 = f_1 * C_0 + i_1 * C̃_1 = [0.73, 0.73] * [0, 0] + [0.73, 0.73] * [0.76, 0.76] = [0.55, 0.55]`
*   **Output Gate:** `o_1 = σ(W_o * [h_0, x_1] + b_o) = σ([[1, 1], [1, 1]] * [0, 0, 0.1, 0.9] + [0, 0]) = σ([1, 1]) = [0.73, 0.73]`
*   **Hidden State:** `h_1 = o_1 * tanh(C_1) = [0.73, 0.73] * tanh([0.55, 0.55]) = [0.73, 0.73] * [0.5, 0.5] = [0.365, 0.365]`

**Time Step 2: Input "love"**

*   `x_2 = [0.8, 0.2]`
*   `h_1 = [0.365, 0.365]`
*   `C_1 = [0.55, 0.55]`

The same calculations are repeated with the new input `x_2` and the previous hidden and cell states `h_1` and `C_1`. The gates will produce new values, and the cell and hidden states will be updated.

**Subsequent Time Steps**

The process continues for each word in the sequence. At each time step, the LSTM takes the embedding of the current word and the hidden state from the previous time step as input. It then uses its gates to update its internal cell state and produce a new hidden state.

This allows the LSTM to build up a representation of the entire sequence, with the cell state holding the long-term memory and the hidden state holding the short-term memory.


### The Forget and Keep Mechanism

The decision to forget or keep information is made by the **forget gate**. The output of the forget gate, `f_t`, is a vector of numbers between 0 and 1. When this vector is multiplied by the previous cell state, `C_{t-1}`, it determines how much of the old information is retained.

In our example, "I love to play", the subject of the sentence is "I". As the LSTM processes the sentence, the information about the subject "I" is important for predicting the next word. Therefore, the forget gate will likely output values close to 1 for the dimensions of the cell state that encode the subject. This ensures that the information about "I" is preserved throughout the sequence.

Let's consider another example: "I love to play, but my brother prefers to read." When the LSTM processes the word "brother", the subject of the sentence changes. At this point, the forget gate might output values close to 0 for the dimensions of the cell state that encode the previous subject "I". This would effectively "forget" the old subject and allow the LSTM to focus on the new subject, "brother".

The calculation for the forget gate at the time step when the input is "brother" would be:

`f_t = σ(W_f * [h_{t-1}, E("brother")] + b_f)`

Here, `h_{t-1}` is the hidden state from the previous time step (which contains information about "I love to play, but my"). The weights `W_f` and bias `b_f` are learned during training. The network learns to recognize that when a new subject like "brother" is introduced, the previous subject "I" is less relevant for the next part of the sentence. As a result, the forget gate `f_t` will have values close to 0 for the dimensions in the cell state that represent the subject "I", and values close to 1 for the dimensions that should be kept.

This ability to selectively forget and keep information is what allows LSTMs to handle long-term dependencies and complex grammatical structures.

**Note:** The decision to forget or keep information is not solely based on the embedding vector of the current word. It is a learned behavior based on the combination of the current input (embedding vector) and the previous hidden state. The network learns to recognize patterns in the sequence of embeddings and hidden states to make these decisions. The embedding vector provides the semantic meaning of the current word, and the hidden state provides the context from the previous time steps. The gates use both of these to decide what to do with the cell state.

After processing the entire sequence, the final hidden state `h_4` is a representation of the entire sentence "I love to play". This hidden state can then be used to predict the next word. For example, we can pass it through a fully connected layer with a softmax activation function to get a probability distribution over the vocabulary. The word with the highest probability (e.g., "football", "games") would be our prediction.

### PyTorch Example

Here is a simple example of how to build an LSTM for sentence prediction in PyTorch.

'''python
import torch
import torch.nn as nn

class SentencePredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentencePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # We take the output from the last time step
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

vocab_size = 1000  # Size of our vocabulary
embedding_dim = 128
hidden_dim = 256
output_dim = vocab_size # Predicting the next word

model = SentencePredictor(vocab_size, embedding_dim, hidden_dim, output_dim)

input_sentence = torch.randint(0, vocab_size, (1, 4))  # Example input sentence (batch of 1, sequence of 4 words)


output = model(input_sentence)
print(output.shape) # torch.Size([1, 1000])
'''

## Pros and Cons

### Pros

*   **Long-Term Dependencies:** LSTMs are very good at learning long-term dependencies.
*   **No Vanishing Gradient:** LSTMs do not suffer from the vanishing gradient problem.
*   **Flexibility:** LSTMs can be used for a wide variety of tasks, including sequence-to-sequence learning, sequence classification, and sequence generation.

### Cons

*   **Complexity:** LSTMs are more complex than standard RNNs, which can make them more difficult to train and understand.
*   **Computational Cost:** LSTMs are computationally expensive, especially for long sequences.
*   **Overfitting:** LSTMs can be prone to overfitting, especially on small datasets.

## Real-World Applications

LSTMs are used in a wide variety of real-world applications, including:

*   **Natural Language Processing (NLP):** Machine translation, text summarization, sentiment analysis, and speech recognition.
*   **Time Series Analysis:** Stock price prediction, weather forecasting, and traffic prediction.
*   **Image Captioning:** Generating a textual description of an image.
*   **Video Analysis:** Classifying the content of a video.

## References

*   [Understanding LSTM Networks -- colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
*   [Long short-term memory - Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
*   [A Critical Review of Recurrent Neural Networks for Sequence Learning](https://arxiv.org/abs/1506.00019)

# Gated Recurrent Unit (GRU)

## Introduction

The Gated Recurrent Unit (GRU) is another type of recurrent neural network that, like LSTM, is designed to solve the vanishing gradient problem that affects standard RNNs. It was introduced by Kyunghyun Cho et al. in 2014. GRUs are a slightly simpler and more computationally efficient variation of LSTMs. They have fewer gates and do not have a separate cell state, which makes them faster to train.

## Architecture and Formulas

The GRU has two gates: a reset gate and an update gate. These gates control the flow of information, deciding what to keep from the past and what new information to add.

![GRU Architecture](https://www.researchgate.net/profile/Yuan-Sun-107/publication/334921339/figure/fig2/AS:788241349382144@1564942349949/The-architecture-of-a-gated-recurrent-unit-GRU.png)

The GRU cell has two main gates:

*   **Reset Gate:** Decides how much of the past information to forget.
*   **Update Gate:** Decides what information to throw away and what new information to add.

### Reset Gate

The reset gate is responsible for determining how to combine the new input with the previous memory. It decides how much of the past information is relevant to the current context.

The formula for the reset gate is:

`r_t = σ(W_r * [h_{t-1}, x_t] + b_r)`

### Update Gate

The update gate is similar to the forget and input gates of an LSTM. It decides what information from the previous hidden state to keep and what new information to add from the current input.

The formula for the update gate is:

`z_t = σ(W_z * [h_{t-1}, x_t] + b_z)`

### Candidate Hidden State

The candidate hidden state is calculated using the reset gate. If the reset gate output for a dimension is close to 0, it ignores the previous hidden state information for that dimension and focuses only on the current input.

`h̃_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)`

### Final Hidden State

The final hidden state is a linear interpolation between the previous hidden state `h_{t-1}` and the candidate hidden state `h̃_t`. The update gate `z_t` controls this interpolation. If `z_t` is close to 1, the new hidden state is mostly the candidate hidden state. If it's close to 0, the new hidden state is mostly the previous hidden state.

`h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t`

## Differences with LSTM

| Feature | LSTM | GRU |
| :--- | :--- | :--- |
| **Gates** | Three gates: Forget, Input, Output | Two gates: Reset, Update |
| **Cell State** | Has a separate cell state (`C_t`) for long-term memory | No separate cell state; merges cell state and hidden state |
| **Complexity** | More complex, more parameters | Simpler, fewer parameters |
| **Computational Cost**| Higher | Lower |

The main difference is that GRU combines the forget and input gates into a single update gate. It also merges the cell state and hidden state. This makes the GRU model simpler than LSTM.

## Use Cases and Benefits

*   **Computational Efficiency:** With fewer parameters, GRUs are faster to train and require less data to generalize.
*   **Performance:** For many tasks, especially with smaller datasets, GRUs can perform on par with LSTMs.
*   **Good Starting Point:** Due to their simplicity, GRUs are often a good first choice before trying more complex models like LSTMs.

GRUs are a good choice when computational resources are limited or when the dataset is not very large. For tasks where you need to capture very long-term dependencies, LSTMs might have a slight edge due to their separate cell state.

## PyTorch Example

Here is a simple example of how to build a GRU for sentence prediction in PyTorch.

'''python
import torch
import torch.nn as nn

class SentencePredictorGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentencePredictorGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        # We take the output from the last time step
        last_hidden_state = gru_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output


vocab_size = 1000  # Size of our vocabulary
embedding_dim = 128
hidden_dim = 256
output_dim = vocab_size # Predicting the next word

model = SentencePredictorGRU(vocab_size, embedding_dim, hidden_dim, output_dim)

input_sentence = torch.randint(0, vocab_size, (1, 4))  # Example input sentence (batch of 1, sequence of 4 words)

output = model(input_sentence)
print(output.shape) # torch.Size([1, 1000])
'''

## References

*   [Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.](https://arxiv.org/abs/1406.1078)
*   [Illustrated Guide to LSTMs and GRUs: A step by step explanation](https://www.analyticsvidhya.com/blog/2021/03/illustrated-guide-to-lstms-and-gru-a-step-by-step-explanation/)
*   [Understanding GRU Networks -- colah's blog (similar concepts to LSTM)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)