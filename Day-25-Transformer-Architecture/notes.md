# Day 25 Notes — Transformer Architecture

## Core idea

A Transformer is more than attention. A simplified block combines:

```text
Multi-Head Attention
        ↓
Residual + Normalization
        ↓
Feed-Forward Network
        ↓
Residual + Normalization
```

## Position

Language is order-sensitive:

```text
dog bites man
man bites dog
```

Positional information gives the model access to token order.

## Attention

From Day 24:

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k))V
```

## Residuals

```text
x → sublayer
└────→ add with x
```

So:

```text
output = x + sublayer(x)
```

## LayerNorm

Layer normalization stabilizes token representations. Transformer variants differ in the exact placement of normalization.

## FFN

```text
x → Linear → activation → Linear → output
```

It adds nonlinear transformation after attention.

## Decoder-only LLMs

GPT-style autoregressive models use causal self-attention:

```text
past/current tokens → visible
future tokens        → blocked
```

## Shape cheat sheet

```text
[B,T,C]
   ↓ split heads
[B,H,T,D]
   ↓ attention
[B,H,T,T]
   ↓ combine
[B,T,C]

C = H × D
```

## Don't confuse

```text
Attention = mechanism
Multi-Head Attention = multiple attention heads
Transformer Block = attention + FFN + residual/norm
Transformer = stack/architecture of blocks
```
