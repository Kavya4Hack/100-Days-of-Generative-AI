# Day 24 — Multi-Head Attention

## Goal

Understand self-attention and multi-head attention from first principles.

Core idea:

> A token can look at other tokens and assign different importance to them.

## Topics

- Why attention is needed
- Query, Key, Value (Q, K, V)
- Attention scores
- Scaling by `sqrt(d_k)`
- Softmax
- Attention weights
- Weighted Values
- Self-attention
- Multi-head attention
- Head splitting and concatenation
- Output projection
- Causal masking
- Transformer connection

## 1. Query, Key, Value

For every input representation, the model creates:

```text
Query (Q)
Key   (K)
Value (V)
```

Useful intuition:

```text
Query → What information am I looking for?
Key   → What information do I contain?
Value → What information should I provide?
```

These are learned projections, not literal questions.

## 2. Attention

The scaled dot-product attention equation is:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / sqrt(d_k))V
```

Step by step:

```text
QKᵀ
 ↓
Similarity scores
 ↓
Scale by sqrt(d_k)
 ↓
Softmax
 ↓
Attention weights
 ↓
Weighted sum of V
 ↓
Attention output
```

## 3. Self-Attention

In self-attention, Q, K and V all come from the same input sequence.

This allows every token to interact with other tokens.

## 4. Multi-Head Attention

Instead of one attention operation, Transformers use multiple attention heads:

```text
Input
 ├── Head 1
 ├── Head 2
 ├── Head 3
 └── Head 4
       ↓
   Concatenate
       ↓
 Output Projection
       ↓
     Output
```

Each head has its own learned Q/K/V projections.

Different heads can learn different relationships. The model learns what those relationships are; they are not manually assigned.

## 5. Why Scale?

Raw dot products can grow as vector dimension grows.

Therefore:

```text
QKᵀ / sqrt(d_k)
```

is used before softmax to keep values in a more manageable range.

## 6. Softmax

Softmax converts scores into normalized weights.

Conceptually:

```text
scores → softmax → weights
```

The weights across the attended positions sum to 1.

## 7. Causal Attention

Autoregressive LLMs such as GPT-style models cannot use future tokens when predicting the next token.

For:

```text
I love machine learning
```

a position must not look ahead at future positions.

A causal mask hides those future positions.

## 8. Important Tensor Shapes

Let:

```text
B = batch size
T = sequence length
C = embedding dimension
H = number of heads
D = head dimension
```

Then:

```text
Input       → [B, T, C]
Split heads → [B, H, T, D]
Scores      → [B, H, T, T]
Output      → [B, T, C]
```

where:

```text
C = H × D
```

Example:

```text
B = 2
T = 5
C = 8
H = 2
D = 4
```

So:

```text
Input       → [2, 5, 8]
Split heads → [2, 2, 5, 4]
Scores      → [2, 2, 5, 5]
Output      → [2, 5, 8]
```

## 9. Transformer Connection

You now have:

```text
Text
 ↓
Tokenization
 ↓
Token IDs
 ↓
Embeddings
 ↓
Self-Attention
 ↓
Multi-Head Attention
 ↓
Feed-Forward Network
 ↓
Transformer Blocks
 ↓
LLM
```

## Run

```bash
python attention.py
```

The implementation demonstrates scaled dot-product attention, causal masking, multi-head attention, and tensor shapes.

## Revision Questions

1. Why do we need attention?
2. What are Query, Key and Value?
3. How is an attention score calculated?
4. Why divide by `sqrt(d_k)`?
5. What does softmax do?
6. What is self-attention?
7. Why use multiple heads?
8. What happens when heads are concatenated?
9. Why is an output projection used?
10. What is causal masking?
11. Why can't an autoregressive model see future tokens?
12. Explain `[B,T,C]` and `[B,H,T,D]`.

## Next

Continue toward the full **Transformer architecture**: positional information, feed-forward networks, residual connections, normalization, encoder/decoder structure, and eventually LLM architecture.
