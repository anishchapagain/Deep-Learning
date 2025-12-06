# Natural Language Processing (NLP)

## What is NLP?
- **Definition**: Computational techniques for analyzing, understanding, and generating human language.
- **Goal**: Bridge the gap between *raw text* and *machine‑readable representations*.
- **Typical Tasks**:
  - Text classification (sentiment, topic)
  - Sequence labeling (POS, NER)
  - Machine translation
  - Question answering
  - Summarisation

## Core NLP Pipeline
1. **Text Acquisition** – raw documents, web crawls, speech‑to‑text.
2. **Pre‑processing** – tokenisation, lower‑casing, stop‑word removal, stemming/lemmatisation.
3. **Feature Extraction** – bag‑of‑words, TF‑IDF, n‑grams.
4. **Embedding Layer** – dense vector representation of tokens.
5. **Modeling** – classic ML (SVM, Naïve Bayes) or deep models (RNN, Transformer).
6. **Post‑processing** – decoding, detokenisation, evaluation.

## Why Word Embeddings?
- **Sparse vs. Dense**: Traditional one‑hot vectors are high‑dimensional and orthogonal → no notion of similarity.
- **Semantic Geometry**: Embeddings map words to a *continuous* space where distance ↔ semantic similarity.
- **Transferability**: Pre‑trained embeddings (Word2Vec, GloVe, fastText) capture world knowledge and can be fine‑tuned.
- **Downstream Benefits**: Faster convergence, better generalisation, enable analogies (e.g., *king – man + woman ≈ queen*).

## Popular Embedding Techniques
| Method | Training Objective | Key Property |
|--------|-------------------|--------------|
| **Word2Vec (CBOW / Skip‑gram)** | Predict surrounding words (or target from context) | Captures local co‑occurrence |
| **GloVe** | Factorise global word‑co‑occurrence matrix | Balances local and global statistics |
| **fastText** | Sub‑word (character n‑gram) embeddings | Handles OOV words |
| **ELMo** | Contextualised embeddings from bi‑directional LSTM | Word meaning varies with context |
| **BERT / Transformer‑based** | Masked language modelling + next‑sentence prediction | Deep contextualisation, bidirectional |


## Embedding Workflow (Step‑by‑Step)
1. **Collect Corpus** – large, domain‑relevant text.
2. **Tokenise & Clean** – consistent tokenisation (e.g., WordPiece for BERT).
3. **Build Vocabulary** – limit size (e.g., top‑50k tokens) and assign IDs.
4. **Choose Embedding Type** – static (Word2Vec) vs. contextual (BERT).
5. **Train / Load** –
   - *Training*: use `gensim` or `torchtext` for Word2Vec/GloVe.
   - *Loading*: `transformers` library for pre‑trained BERT.
6. **Integrate** – replace one‑hot vectors with embedding lookup in your model.
7. **Fine‑Tune (optional)** – continue training on downstream task data.


## Practical Tips for Teaching Slides
- **Visualise**: 2‑D t‑SNE / PCA plot of embeddings to show clusters.
- **Analogy Demo**: Compute vector arithmetic for *king – man + woman*.
- **Code Snippet** (Python, PyTorch):
```python
import torch
import torch.nn as nn

# Example: static embedding layer
vocab_size = 10000
embed_dim = 300
embedding = nn.Embedding(vocab_size, embed_dim)

# Lookup embedding for token IDs
ids = torch.tensor([12, 45, 78])  # example token IDs
vectors = embedding(ids)  # shape: (3, embed_dim)
```
- **Discussion**: When to freeze vs. fine‑tune embeddings.
- **Pitfalls**: OOV handling, bias in pre‑trained vectors, dimensionality trade‑off.


## Beyond Word‑Level – Advanced Topics
- **Sub‑word / Character Embeddings** – useful for morphologically rich languages.
- **Sentence / Document Embeddings** – averaging, Doc2Vec, Universal Sentence Encoder.
- **Contextual Transformers** – BERT, RoBERTa, GPT – shift from static to dynamic representations.
- **Multilingual Embeddings** – align spaces across languages (MUSE, XLM‑R).

## Summary
- NLP transforms text into structured data for machines.
- Word embeddings are the *core bridge* that encode semantic meaning.
- Choose embedding strategy based on data size, domain, and task complexity.
- Hands‑on demos and visualisations cement understanding.