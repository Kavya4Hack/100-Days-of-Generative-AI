# Day 26 Notes — BERT vs GPT vs Encoder-Decoder

## BERT

```text
Encoder-only
↓
Bidirectional context
↓
Masked Language Modeling
↓
Understanding
```

Example:

```text
The dog is [MASK] outside.
```

The model predicts the masked token using surrounding context.

## GPT

```text
Decoder-only
↓
Causal self-attention
↓
Next-token prediction
↓
Generation
```

Future tokens are hidden.

## T5

```text
Encoder + Decoder
↓
Text → Text
↓
Sequence-to-sequence
```

Examples:

```text
English → French
article → summary
question → answer
```

## Cross-Attention

```text
Q = decoder
K,V = encoder output
```

The decoder can use information produced by the encoder.

## Comparison

```text
BERT → Encoder-only → Understand → Masked tokens
GPT  → Decoder-only → Generate   → Next token
T5   → Encoder+Decoder → Transform → Text to text
```

## Important distinction

```text
Self-attention
= attention within the same representation source

Cross-attention
= one representation source attends to another
```

## Common mistakes

- BERT is not a decoder-only generative model.
- GPT-style causal generation cannot see future tokens.
- T5 is not simply "GPT + BERT"; it is an encoder-decoder Transformer.
- Architecture labels are useful mental models, not rigid descriptions of every modern model.
