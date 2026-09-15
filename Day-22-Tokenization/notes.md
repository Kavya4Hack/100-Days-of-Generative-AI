# Day 22 Notes — Tokenization

## 1. What is Tokenization?

Tokenization means breaking text into smaller pieces called tokens.

```text
"I love AI"
→ ["I", "love", "AI"]
```

A token can be:

- A word
- A character
- A subword
- Sometimes a byte-level unit

---

## 2. Why Not Just Use Words?

Suppose the model has seen:

```text
play
playing
played
player
```

Keeping every complete word as a separate vocabulary item can make the vocabulary huge.

Subword tokenization can instead reuse pieces:

```text
play + ing
play + ed
play + er
```

This makes vocabulary more efficient.

---

## 3. Word vs Character vs Subword

| Method | Example | Main Advantage | Main Problem |
|---|---|---|---|
| Word | `["I", "love"]` | Easy to understand | Huge vocabulary / unknown words |
| Character | `["I", "l", "o", "v", "e"]` | Very flexible | Very long sequences |
| Subword | `["lov", "e"]` | Good balance | More complex |

---

## 4. BPE

Byte Pair Encoding repeatedly merges frequently occurring pairs.

The important idea is:

```text
small pieces
     ↓
frequent combinations
     ↓
useful subword tokens
```

You do not need to implement a production BPE tokenizer yet. Understand the mechanism first.

---

## 5. WordPiece

WordPiece is another subword approach.

Important association:

```text
BERT → WordPiece
```

Do not assume every Transformer model uses WordPiece.

---

## 6. Unigram

Unigram chooses a likely segmentation from a vocabulary of subword pieces.

It is commonly associated with SentencePiece tokenization systems.

---

## 7. Special Tokens

Special tokens provide structural information.

Examples:

```text
[CLS]
[SEP]
[PAD]
[UNK]
<BOS>
<EOS>
```

Different models use different conventions.

---

## 8. Token IDs

A tokenizer maintains a vocabulary.

Conceptually:

```text
"hello" → 1532
"world" → 2941
```

The IDs themselves have no inherent meaning.

The model learns useful representations from these IDs through embeddings.

---

## 9. Important Distinction

### Token

A piece of text.

```text
"hello"
```

### Token ID

A number representing that token in a particular vocabulary.

```text
1532
```

### Embedding

A learned vector representing the token.

Conceptually:

```text
1532
 ↓
[0.21, -0.43, 0.87, ...]
```

Embeddings are the topic for the next stage.

---

## 10. LLM Pipeline

Memorize this:

```text
Text
 ↓
Tokenizer
 ↓
Tokens
 ↓
Token IDs
 ↓
Embedding
 ↓
Transformer
 ↓
Output Token
 ↓
More Tokens
 ↓
Generated Text
```

---

## ⚠️ Common Beginner Mistakes

### Mistake 1

Thinking:

```text
1 word = 1 token
```

Not necessarily.

### Mistake 2

Thinking token IDs contain meaning.

They are vocabulary indexes.

### Mistake 3

Thinking all LLMs use the same tokenizer.

Different models can use different tokenization algorithms and vocabularies.

### Mistake 4

Thinking tokenization is only splitting by spaces.

Modern tokenizers are much more sophisticated.

---

## 📝 Quick Revision

Before moving to Day 23, make sure you can answer:

1. What is tokenization?
2. What is the difference between word and character tokenization?
3. Why is subword tokenization useful?
4. What is BPE?
5. What is WordPiece?
6. What is Unigram?
7. What are special tokens?
8. What is a token ID?
9. What is the difference between a token and an embedding?
10. Why does tokenization affect LLM cost and context length?
