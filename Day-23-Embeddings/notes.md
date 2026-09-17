# Day 23 Notes — Embeddings

## 1. Definition

An embedding is a dense numerical vector used to represent information such as a token, word, sentence, document, or other data.

## 2. Token ID vs Embedding

```text
"cat" → 1250
```

`1250` is simply a vocabulary index.

The embedding layer maps it to something like:

```text
1250 → [0.21, -0.43, 0.71, ...]
```

The vector is the learned representation.

## 3. Embedding Matrix

If:

```text
vocabulary size = 5
embedding dimension = 3
```

the embedding matrix has shape:

```text
(5, 3)
```

Each row corresponds to one token.

## 4. One-Hot vs Embedding

One-hot vectors identify an item but do not naturally encode semantic relationships.

Embeddings learn useful relationships in a continuous vector space.

## 5. Cosine Similarity

```text
cos(A,B) = (A · B) / (||A|| ||B||)
```

It measures how similarly two vectors point.

Common use:

```text
query embedding
      ↓
compare with document embeddings
      ↓
retrieve similar documents
```

## 6. Static vs Contextual

Classic Word2Vec-style embeddings generally give a word one learned representation.

Transformer representations can depend on context:

```text
river bank
bank account
```

## 7. Sentence and Document Embeddings

A complete sentence/document can be converted to one vector.

This enables:

- semantic search
- clustering
- similarity detection
- retrieval
- recommendation systems

## 8. RAG Connection

```text
Document → Chunks → Embeddings → Vector DB
Query → Embedding → Similarity Search → Relevant Context → LLM
```

This is one of the most important connections between embeddings and Generative AI.

## ⚠️ Common Mistakes

- Token ID is not the same thing as an embedding.
- Embedding values are not manually assigned meanings.
- Higher similarity does not guarantee truth.
- Different models can produce different embedding spaces.
- Not every model uses the same embedding dimension.

## 🧠 Memory Shortcut

```text
TOKEN
 ↓
TOKEN ID
 ↓
EMBEDDING
 ↓
CONTEXTUAL PROCESSING
 ↓
TRANSFORMER
```
