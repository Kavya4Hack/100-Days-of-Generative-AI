# Day 23 — Embeddings

## 🎯 Goal
Understand how tokens are converted into dense numerical vectors called **embeddings**, how semantic similarity works, and why embeddings are fundamental to semantic search and RAG.

## 📚 Topics Covered
- What is an embedding?
- Token IDs vs embeddings
- Sparse vs dense representations
- Embedding vectors and dimensions
- Embedding matrices
- Cosine similarity
- Word, sentence, and document embeddings
- Static vs contextual representations
- Semantic search
- Embeddings in RAG
- Text → tokens → IDs → embeddings → Transformer

## 🧠 Core Idea

A token ID is an index:

```text
"cat" → 1523
```

An embedding converts that ID into a learned vector:

```text
1523 → [0.21, -0.42, 0.73, ...]
```

The ID itself has no semantic meaning. The learned vector is the useful representation.

## 📊 Sparse vs Dense

One-hot encoding:

```text
cat → [1, 0, 0, 0]
dog → [0, 1, 0, 0]
car → [0, 0, 1, 0]
```

Embeddings are dense vectors:

```text
cat → [0.21, -0.42, 0.73, 0.15]
dog → [0.19, -0.39, 0.70, 0.17]
```

The values above are illustrative. A trained model learns its own representations.

## 📐 Embedding Dimension

If an embedding has dimension 768, each item is represented by 768 numerical values.

An embedding matrix with vocabulary size 10,000 and dimension 128 has shape:

```text
(10000, 128)
```

A token ID selects one row from this matrix.

## 📏 Cosine Similarity

Cosine similarity compares the direction of two vectors:

```text
cos(A,B) = (A · B) / (||A|| ||B||)
```

It is commonly used in semantic search.

A higher similarity indicates closer representation in that embedding space; it does not prove factual correctness.

## 🌍 Types of Embeddings

### Word embeddings
Represent individual words. Classic examples include Word2Vec and GloVe.

### Sentence embeddings
Represent an entire sentence and are useful for similarity, clustering, and search.

### Document embeddings
Represent larger text units and are useful in retrieval systems.

## 🧠 Static vs Contextual Representations

Classic word embeddings generally associate one learned vector with a word.

Transformer-based representations can be influenced by surrounding context:

```text
river bank
bank account
```

The representation can differ because the context differs.

## 🔎 Embeddings in Semantic Search

Instead of relying only on exact keyword matching:

```text
Query:
"How do I recover my account?"

Document:
"Steps for resetting your login credentials"
```

An embedding model can represent both texts in the same vector space and allow semantic comparison.

## 🤖 Embeddings in RAG

```text
Documents
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vectors
   ↓
Vector Database

User Query
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Chunks
   ↓
LLM
   ↓
Answer
```

This is a key foundation for later RAG work.

## 🛠️ Run

```bash
python embeddings.py
```

## 🔑 Key Takeaways

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
```

And:

```text
Embedding = numerical representation
Semantic Search = similarity between representations
```

## 📝 Revision Questions

1. What is an embedding?
2. What is the difference between a token ID and an embedding?
3. Why are embeddings dense?
4. What does embedding dimension mean?
5. What is cosine similarity?
6. What is an embedding matrix?
7. What are sentence and document embeddings?
8. What is a contextual representation?
9. How are embeddings used in semantic search?
10. How are embeddings used in RAG?

## 🚀 Next Step

**Day 24 — Transformers**

You now understand:

```text
Text → Tokens → Token IDs → Embeddings
```

Next: how Transformers use these representations and self-attention to understand relationships between tokens.
