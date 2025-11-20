# Natural Language Processing: Word Embeddings and Semantic Meaning

## 1. Introduction to NLP

Natural Language Processing (NLP) is a field of artificial intelligence (AI) that enables computers to understand, interpret, and generate human language. From translation services and chatbots to sentiment analysis and spam filters, NLP is the driving force behind many applications we use daily.

A fundamental challenge in NLP is converting text—which is unstructured and categorical—into a numerical format that machine learning models can process.

## 2. The Problem with Traditional Representations

How can we represent a word like "cat"?

A simple approach is **One-Hot Encoding**. We create a vector with the same length as our entire vocabulary. The vector is all zeros except for a single '1' at the index corresponding to our word.

**Vocabulary:** `[apple, ball, cat, dog, ...]`

- `apple`: `[1, 0, 0, 0, ...]`
- `ball`:  `[0, 1, 0, 0, ...]`
- `cat`:   `[0, 0, 1, 0, ...]`
- `dog`:   `[0, 0, 0, 1, ...]`

**Problems with this approach:**

1.  **Huge & Sparse Vectors:** For a vocabulary of 50,000 words, each word is a 50,000-dimensional vector, which is computationally inefficient.
2.  **No Semantic Relationship:** The one-hot vectors for "cat" and "dog" are mathematically as different from each other as "cat" and "apple". The representation doesn't capture the fact that cats and dogs are both animals and are more related to each other than to a fruit.

## 3. Word Embeddings: Representing Words by Their Company

Word Embeddings solve the problems of one-hot encoding by representing words as dense, low-dimensional vectors. This approach is grounded in the **Distributional Hypothesis**, a core linguistic theory that states: **"a word is characterized by the company it keeps."**

In other words, words that appear in similar contexts (e.g., "The cat sat on the...") are likely to have similar meanings. Word embedding algorithms are designed to capture these contextual similarities and encode them into dense vectors.

**Key Idea:** The goal is to create a high-dimensional space where vectors of semantically similar words are close to each other.

This allows models to understand nuanced relationships. For example, a famous demonstration is the vector arithmetic:

`vector('King') - vector('Man') + vector('Woman') ≈ vector('Queen')`

This shows that the model has learned abstract concepts like gender and royalty purely from observing word patterns in text.

## 4. How Word Embeddings are Trained

Word embeddings are not created manually; they are the **byproduct of a "fake" predictive task**. We train a simple neural network on a large corpus of text to predict something about word context. The actual prediction is discarded, but the learned weights of a specific hidden layer in the network become our word embeddings.

The most famous algorithms for this are from the **Word2Vec** family (developed at Google) and **GloVe** (developed at Stanford).

### 4.1 Word2Vec: Predictive Models

Word2Vec uses a shallow neural network and comes in two main flavors.

#### A. Continuous Bag-of-Words (CBOW)

-   **Core Idea:** Predict a target word using its surrounding context words. It’s like a fill-in-the-blank task.
-   **Example:** Given the context `("a", "quick", "fox", "jumps")`, the model's goal is to predict the target word `brown`.
-   **How it Works (Simplified):**
    1.  The context words (`"a"`, `"quick"`, `"fox"`, `"jumps"`) are fed into the network as one-hot vectors.
    2.  The network looks up their corresponding embedding vectors from a large, randomly initialized **embedding matrix**. This matrix has a row for every word in the vocabulary and a column for each dimension of the embedding (e.g., 300).
    3.  The embedding vectors of the context words are averaged to create a single context vector.
    4.  This context vector is fed through a final layer to produce a probability distribution over the entire vocabulary.
    5.  The model calculates the error between its prediction and the actual target word (`brown`) and uses backpropagation to update the network's weights.
    6.  **Crucially, the weights in the embedding matrix are also updated.** To get better at predicting `brown`, the model must improve the embedding vectors for the context words. Over millions of examples, words that appear in similar contexts receive similar updates, causing their vectors to converge in the embedding space.

#### B. Skip-Gram

-   **Core Idea:** The opposite of CBOW. Given a single input word, predict its surrounding context words.
-   **Example:** Given the input word `brown`, the model tries to predict words like `("a", "quick", "fox", "jumps")`.
-   **How it Works (Simplified):**
    1.  The input word (`brown`) is fed into the network.
    2.  Its embedding is looked up from the embedding matrix.
    3.  This single embedding is used to predict multiple context words.
    4.  The error between the predictions and the actual context words is calculated, and the weights are updated via backpropagation.
-   **Comparison:** Skip-gram is generally slower to train than CBOW but is considered more effective at capturing the meaning of rare words and results in better overall semantic representations.

### 4.2 GloVe: Global Vectors

GloVe (Global Vectors for Word Representation) takes a different approach. It argues that global co-occurrence statistics hold the key to meaning.

-   **Core Idea:** Learn embeddings by directly factorizing a matrix of word co-occurrence statistics.
-   **How it Works (Simplified):**
    1.  **Build a Co-occurrence Matrix:** First, the algorithm makes a single pass over the entire corpus to build a large matrix `X`, where `X[i, j]` counts how many times word `j` has appeared in the context of word `i`.
    2.  **Learn Vectors to Recapture Ratios:** GloVe then trains a model whose objective is to learn word vectors such that their dot product `vector(i) · vector(j)` is proportional to the logarithm of their probability of co-occurrence `log(P(i,j))`.
-   **Intuition:** By focusing on the ratios of co-occurrence probabilities rather than the raw probabilities themselves, GloVe can capture finer-grained relationships between words more effectively and often trains faster than Word2Vec.

### Summary of Training Models

| Model     | Core Idea                               | Pros                                       | Cons                               |
|-----------|-----------------------------------------|--------------------------------------------|------------------------------------|
| **CBOW**  | Predict center word from context.       | Fast to train, good for frequent words.    | Less effective for rare words.     |
| **Skip-Gram**| Predict context words from center word. | Excellent for rare words, captures fine-grained semantic relationships. | Slower to train.                   |
| **GloVe** | Use co-occurrence stats to learn vectors. | Fast, leverages global statistics.         | Can require significant memory for the co-occurrence matrix. |

## 5. Measuring Semantic Similarity: Cosine Similarity

Once words are represented as dense vectors (embeddings), we need a way to quantify how "similar" two words are in this vector space. This is where **Cosine Similarity** comes in.

**Semantic Similarity** refers to the degree to which two words or phrases are related in meaning. In the context of word embeddings, if two words are semantically similar, their embedding vectors should point in roughly the same direction in the high-dimensional space.

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two non-zero vectors. The closer the cosine value is to 1, the smaller the angle, and thus the more similar the vectors (and the words they represent). A value of 0 indicates orthogonality (no similarity), and -1 indicates complete dissimilarity (opposite directions).

The formula for cosine similarity between two vectors, A and B, is:

\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}} 

Where:
-   $A \cdot B$ is the dot product of vectors A and B.
-   $\|A\|$ and $\|B\|$ are the Euclidean magnitudes (L2 norms) of vectors A and B, respectively.

### Why Cosine Similarity for Embeddings?

-   **Direction over Magnitude:** Cosine similarity focuses on the orientation of the vectors rather than their magnitudes. This is crucial for word embeddings because the length of an embedding vector might not always be directly indicative of its meaning, but its direction relative to other vectors often is.
-   **Range:** The output is always between -1 and 1, making it easy to interpret.

### Code Example: Calculating Cosine Similarity in PyTorch

Let's use our previously generated embeddings to calculate their similarity.

```python
import torch
import torch.nn as nn # Needed for nn.Embedding in the dummy setup
import torch.nn.functional as F

# Assume we have our embedding_layer and vocab from previous examples
# For demonstration, let's create some dummy embeddings for 'cat', 'dog', and 'apple'
# In a real scenario, these would come from your trained or pre-trained embedding_layer
embedding_dim = 10
# Let's manually set some vectors for illustration
# These are just random for now, but imagine they are learned vectors
# We'll make 'cat' and 'dog' somewhat similar, 'cat' and 'apple' dissimilar,
# and 'cat' and 'feline' very similar.
cat_vec = torch.tensor([0.8, 0.7, 0.1, 0.2, 0.9, 0.6, 0.3, 0.4, 0.5, 0.7], dtype=torch.float32)
dog_vec = torch.tensor([0.7, 0.8, 0.2, 0.1, 0.8, 0.7, 0.4, 0.3, 0.6, 0.8], dtype=torch.float32)
apple_vec = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.7, 0.6, 0.2, 0.1], dtype=torch.float32)
feline_vec = torch.tensor([0.85, 0.75, 0.15, 0.25, 0.95, 0.65, 0.35, 0.45, 0.55, 0.75], dtype=torch.float32)


# Function to calculate cosine similarity
def calculate_cosine_similarity(vec1, vec2):
    # F.cosine_similarity expects inputs of shape (N, D) or (D,)
    # If they are (D,), it will treat them as (1, D)
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)) # Add unsqueeze for batch dimension

# Calculate similarities
sim_cat_dog = calculate_cosine_similarity(cat_vec, dog_vec)
sim_cat_apple = calculate_cosine_similarity(cat_vec, apple_vec)
sim_cat_feline = calculate_cosine_similarity(cat_vec, feline_vec)

print(f"Cosine Similarity between 'cat' and 'dog': {sim_cat_dog.item():.4f}")
print(f"Cosine Similarity between 'cat' and 'apple': {sim_cat_apple.item():.4f}")
print(f"Cosine Similarity between 'cat' and 'feline': {sim_cat_feline.item():.4f}")

# Expected output (with actual learned embeddings):
# 'cat' and 'dog' should have a high similarity score (e.g., > 0.7)
# 'cat' and 'apple' should have a low similarity score (e.g., < 0.3)
# 'cat' and 'feline' should have a very high similarity score (e.g., > 0.9)
```

## 6. Using Embeddings in PyTorch

PyTorch provides a simple way to manage embeddings through the `nn.Embedding` layer. This layer is essentially a lookup table that stores and retrieves embeddings for a fixed vocabulary. When you create an `nn.Embedding` layer, you are creating the **embedding matrix** described above.

```python
import torch
import torch.nn as nn

# 1. Define our vocabulary and map words to indices
vocab = {'hello': 0, 'world': 1, 'pytorch': 2, 'is': 3, 'awesome': 4}
vocab_size = len(vocab)
embedding_dim = 10 # The desired vector size for each word

# 2. Create the embedding layer (the embedding matrix)
# This initializes a matrix of size (vocab_size, embedding_dim) with random weights.
# These weights are what get "learned" during model training, just like in Word2Vec.
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 3. Get the embedding for a word
# To get a word's vector, we pass its index to the layer.
word_to_lookup = 'pytorch'
word_index = vocab[word_to_lookup]
word_tensor = torch.tensor([word_index], dtype=torch.long)

pytorch_embedding = embedding_layer(word_tensor)

print(f"--- Embedding for '{word_to_lookup}' ---")
print(pytorch_embedding)
```
When this `embedding_layer` is part of a larger network for a task like text classification, the gradients from the loss function flow all the way back and update the rows of the embedding matrix, thus fine-tuning the word vectors for that specific task.

## 7. Using Pre-trained Word Embeddings

Training embeddings from scratch requires a massive amount of data and computational power. Fortunately, researchers have already trained embeddings on huge text corpora (like all of Wikipedia and Google News) and have made them available.

Popular pre-trained models include:
- **Word2Vec**
- **GloVe**
- **FastText**

Using pre-trained embeddings can significantly improve your model's performance, especially if you have a small dataset. It provides your model with a head start by giving it access to general semantic knowledge.

### Example: Loading Pre-trained GloVe Embeddings in PyTorch

Here's a conceptual example of how you would load pre-trained GloVe embeddings into a PyTorch `nn.Embedding` layer.

```python
import torch
import torch.nn as nn
import numpy as np

# Assume we have a function to load GloVe vectors
def load_glove_vectors(glove_path):
    # This is a simplified parser.
    word_to_vector = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            word_to_vector[word] = vector
    return word_to_vector

# --- In your model setup ---

# 1. Your model's vocabulary
vocab = {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4}
vocab_size = len(vocab)
embedding_dim = 100 # Must match the dimension of the GloVe file

# 2. Create a weight matrix for your vocabulary
weight_matrix = np.zeros((vocab_size, embedding_dim))
# A real implementation would load the GloVe file, but we'll simulate it
glove_vectors_simulation = {
    'the': np.random.rand(embedding_dim), 'cat': np.random.rand(embedding_dim),
    'sat': np.random.rand(embedding_dim), 'on': np.random.rand(embedding_dim),
    'mat': np.random.rand(embedding_dim)
}

for word, i in vocab.items():
    if word in glove_vectors_simulation:
        weight_matrix[i] = glove_vectors_simulation[word]

# 3. Create the embedding layer from the pre-trained weights
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.load_state_dict({'weight': torch.from_numpy(weight_matrix)})

# 4. Freeze the embedding layer (optional) to prevent it from changing during training
embedding_layer.weight.requires_grad = False

print("--- Embedding layer created with pre-trained weights ---")
```

## 8. Beyond Static Embeddings: The Transformer Era (BERT & GPT)

While Word2Vec and GloVe are powerful, they have a significant limitation: they generate a single, static embedding for each word. This means the word "bank" has the exact same vector in "river bank" and "investment bank". This fails to capture the rich, context-dependent meaning of words.

This is where **Transformer-based models** like BERT and GPT revolutionize NLP. They create **contextual embeddings**, where the vector for a word is dynamically generated based on the entire sentence it appears in.

### 8.1 The Transformer Architecture: The Engine of Modern NLP

Both BERT and GPT are built upon the **Transformer architecture**. Its key innovation is the **self-attention mechanism**.

**Self-Attention:** This mechanism allows the model to weigh the importance of all other words in the input text when producing the representation for a single word. For "river bank", the attention mechanism would focus on "river" when encoding "bank". For "investment bank", it would focus on "investment". This is how context is dynamically incorporated.

### 8.2 BERT: The Bidirectional Encoder

BERT (Bidirectional Encoder Representations from Transformers) is designed to understand language. It reads an entire text sequence at once, allowing it to learn deep, bidirectional relationships.

**How it Works:**
-   **Bidirectional:** Unlike models that read text left-to-right or right-to-left, BERT processes the whole sentence simultaneously. This gives it a deep understanding of the context from both directions.
-   **Pre-training Tasks:** BERT is pre-trained on two main tasks:
    1.  **Masked Language Model (MLM):** 15% of the words in a sentence are hidden (masked), and the model's job is to predict them based on the surrounding unmasked words. This forces it to learn a rich understanding of language structure and co-occurrence.
    2.  **Next Sentence Prediction (NSP):** The model receives two sentences, A and B, and must predict whether B is the actual sentence that follows A in the original text. This helps it understand relationships between sentences.

**Usage (Fine-Tuning):**
BERT is an excellent **encoder**. You use the pre-trained model and "fine-tune" it on a specific downstream task. This involves adding a small, task-specific layer (e.g., a classification layer) on top of the core BERT model. It excels at tasks requiring deep language understanding:
-   Text Classification (Sentiment Analysis, Topic Categorization)
-   Question Answering
-   Named Entity Recognition (NER)

**Code Example (using Hugging Face `transformers`):**
This example shows how to get contextual embeddings from BERT for a sentence.

```python
# You need to install the transformers library first: pip install transformers torch
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences
sentences = [
    "The cat sat on the mat.",
    "The dog played in the park."
]

# Tokenize the sentences
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Get the model's output
with torch.no_grad():
    outputs = model(**inputs)

# The 'last_hidden_state' contains the contextual embeddings for each token
last_hidden_state = outputs.last_hidden_state

# The embedding for the first token ([CLS]) is often used as a sentence representation
cls_embedding = last_hidden_state[:, 0, :]
```

### 8.3 GPT: The Generative Decoder

GPT (Generative Pre-trained Transformer) is designed to generate human-like text. It is an **auto-regressive** model, meaning it generates text one word at a time, from left to right.

**How it Works:**
-   **Unidirectional (Auto-regressive):** When predicting the next word, GPT can only see the words that came before it. It cannot see future words.
-   **Pre-training Task:** GPT is pre-trained on a single, simple task: **predicting the next word** in a massive corpus of text. By doing this over and over, it learns grammar, facts about the world, and different styles of writing.

**Usage:**
GPT is a powerful **decoder**. It excels at any task that involves generating new text based on a prompt:
-   Text Generation (Story Writing, Code Generation)
-   Summarization
-   Translation
-   Chatbots and Conversational AI

**Code Example (using Hugging Face `transformers`):**
This example shows how to use GPT-2 to generate text.

```python
# You need to install the transformers library first: pip install transformers torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.config.pad_token_id = model.config.eos_token_id

# Encode a prompt
prompt = "Natural Language Processing is a field of AI that"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 8.4 Key Differences: BERT vs. GPT

| Feature        | BERT (Encoder)                                       | GPT (Decoder)                                        |
|----------------|------------------------------------------------------|------------------------------------------------------|
| **Architecture** | **Bidirectional** (sees the whole sentence at once)  | **Unidirectional** (sees only past words)            |
| **Objective**    | Predict masked words (MLM) & next sentence (NSP)     | Predict the next word in a sequence                  |
| **Primary Use**  | **Language Understanding** (NLU)                     | **Language Generation** (NLG)                        |
| **Best For**     | Classification, Q&A, NER                             | Text generation, summarization, translation, chatbots|
| **Analogy**      | An expert who **reads and understands** a document.  | An expert who **writes** a document.                 |


### 8.5 References and Further Reading

-   **The Transformer:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The foundational paper that introduced the Transformer architecture.
-   **BERT:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) - The original BERT paper.
-   **GPT:** [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - The first GPT paper.
-   **Hugging Face:** [Hugging Face Documentation](https://huggingface.co/docs/transformers) - The essential library for working with Transformer models in PyTorch and TensorFlow.