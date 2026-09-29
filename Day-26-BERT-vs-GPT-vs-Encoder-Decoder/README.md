# Day 26 — BERT vs GPT vs Encoder-Decoder

## Goal
Understand the three major Transformer architecture patterns:

```text
BERT → Encoder-only
GPT  → Decoder-only
T5   → Encoder-Decoder
```

## Topics
- BERT and bidirectional context
- Masked Language Modeling (MLM)
- GPT and autoregressive generation
- Causal masking
- Next-token prediction
- T5 and sequence-to-sequence learning
- Cross-attention
- BERT vs GPT vs T5
- Architecture selection by task

## BERT — Encoder Only

```text
Input
 ↓
Embeddings + Position
 ↓
Transformer Encoder Blocks
 ↓
Contextual Representations
 ↓
Task Head
```

BERT can use context from both directions. A major pretraining objective is Masked Language Modeling:

```text
The cat sat on the [MASK].
              ↓
             mat
```

## GPT — Decoder Only

```text
Tokens
 ↓
Embeddings + Position
 ↓
Causal Self-Attention
 ↓
Transformer Blocks
 ↓
Language Model Head
 ↓
Next-token probabilities
```

GPT-style autoregressive models predict the next token using previous/current context while future positions are masked.

```text
"I love" → predict "AI"
"I love AI" → predict next token
```

## T5 — Encoder-Decoder

```text
Input text
 ↓
Encoder
 ↓
Encoder representations
 ↓
Decoder + Cross-Attention
 ↓
Output text
```

T5 treats many NLP tasks as text-to-text transformations:

```text
English → French
article → summary
question + context → answer
```

## Self-Attention vs Cross-Attention

Self-attention: Q, K and V come from the same representation source.

Cross-attention in encoder-decoder models:

```text
Q → decoder
K,V → encoder output
```

## Comparison

| Family | Architecture | Context | Common objective/use |
|---|---|---|---|
| BERT | Encoder-only | Bidirectional | Masked-token learning / understanding |
| GPT | Decoder-only | Causal | Next-token prediction / generation |
| T5 | Encoder-decoder | Encoder + decoder cross-attention | Text-to-text transformation |

These are architecture-level mental models; modern model families can contain variations.

## Run

```bash
python architecture_comparison.py
```

The code is an educational PyTorch demonstration and does not download or train pretrained BERT/GPT/T5 models.

## Revision Questions

1. What is BERT?
2. Why is BERT bidirectional?
3. What is Masked Language Modeling?
4. What is GPT?
5. Why does GPT use causal masking?
6. What is autoregressive generation?
7. What is T5?
8. What is cross-attention?
9. Self-attention vs cross-attention?
10. Compare BERT, GPT and T5.

## Memory Trick

```text
BERT = Understand → Encoder
GPT  = Generate   → Decoder
T5   = Transform  → Encoder + Decoder
```
