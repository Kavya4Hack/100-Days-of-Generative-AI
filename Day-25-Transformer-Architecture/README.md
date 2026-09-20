# Day 25 — Transformer Architecture

## Goal
Understand how token embeddings, positional information, multi-head self-attention, residual connections, LayerNorm, and feed-forward networks combine into a Transformer block.

## Architecture

```text
Token IDs
   ↓
Token Embeddings + Positional Information
   ↓
Multi-Head Self-Attention
   ↓
Residual + LayerNorm
   ↓
Feed-Forward Network
   ↓
Residual + LayerNorm
   ↓
Transformer Block
   ↓
Repeat blocks
```

## Topics
- Why Transformers need positional information
- Multi-head self-attention
- Residual/skip connections
- Layer normalization
- Feed-forward networks
- Transformer blocks
- Encoder vs decoder
- Causal masking
- Tensor shapes

### Positional information

Attention does not inherently encode token order. A common original Transformer approach uses sinusoidal encoding:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Modern architectures may use other positional methods such as learned positions or rotary positional embeddings.

### Feed-forward network

```text
x → Linear → GELU → Linear → output
```

The FFN operates independently at each token position.

### Residual connection

```text
output = x + sublayer(x)
```

Residual connections help information and gradients flow through deep networks.

### Encoder vs Decoder

The original Transformer has an encoder and decoder. Encoder blocks use self-attention. Decoder blocks use masked self-attention and, in encoder-decoder architectures, cross-attention. GPT-style autoregressive LLMs are decoder-only and use causal self-attention.

### Important shapes

```text
B = batch
T = sequence length
C = model dimension
H = number of heads
D = head dimension

Input       [B,T,C]
Split heads [B,H,T,D]
Scores      [B,H,T,T]
Output      [B,T,C]

C = H × D
```

## Run

```bash
python transformer.py
```

The implementation is educational and demonstrates positional encoding, multi-head self-attention, FFN, LayerNorm, residual connections, and stacked Transformer blocks.

## Revision

1. Why is positional information needed?
2. What does multi-head attention do?
3. What is a residual connection?
4. Why use LayerNorm?
5. What does the FFN do?
6. What is a Transformer block?
7. Attention vs Transformer: what is the difference?
8. Encoder vs decoder?
9. Why is causal masking needed for autoregressive generation?
10. Explain `[B,T,C] → [B,H,T,D] → [B,H,T,T] → [B,T,C]`.

## Mental model

```text
Embedding + Position
        ↓
Attention
        ↓
Add + Norm
        ↓
FFN
        ↓
Add + Norm
        ↓
Transformer Block
        ↓
Repeat
```
