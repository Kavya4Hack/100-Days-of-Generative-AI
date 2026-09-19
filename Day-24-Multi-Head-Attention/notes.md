# Day 24 Notes — Multi-Head Attention

## Core idea

Attention lets a token decide which other tokens are relevant to it.

## Q, K, V

```text
Q = Query
K = Key
V = Value
```

Memory shortcut:

```text
Q → Ask
K → Match
V → Information
```

## Main equation

```text
Attention(Q,K,V)
=
softmax(QKᵀ / sqrt(d_k))V
```

Breakdown:

```text
QKᵀ
 ↓
scores
 ↓
scale
 ↓
softmax
 ↓
attention weights
 ↓
weighted V
 ↓
output
```

## Why QKᵀ?

Every query is compared with every key.

If there are 5 tokens, the attention score matrix is:

```text
5 × 5
```

Each row represents how one query position attends across the key positions.

## Self-attention

Q, K and V come from the same input sequence.

That lets tokens interact with other tokens.

## Multi-head attention

Multiple attention heads perform attention in different learned subspaces.

```text
Input
 ↓
Q/K/V projections
 ↓
Split into heads
 ↓
Attention per head
 ↓
Concatenate
 ↓
Output projection
```

If:

```text
d_model = 512
heads = 8
```

then:

```text
head_dim = 512 / 8 = 64
```

## Causal masking

For autoregressive generation:

```text
previous tokens + current token → allowed
future tokens → blocked
```

This prevents future information from leaking into next-token prediction.

## Shape cheat sheet

```text
Input:
[B, T, C]

After splitting heads:
[B, H, T, D]

Attention scores:
[B, H, T, T]

After combining:
[B, T, C]
```

where:

```text
C = H × D
```

## Common mistakes

- Q, K and V are not three separate sentences.
- Attention does not simply choose one word; it creates weights across positions.
- Attention heads are not manually assigned roles.
- Do not forget the `sqrt(d_k)` scaling.
- Causal attention must hide future tokens for autoregressive generation.
