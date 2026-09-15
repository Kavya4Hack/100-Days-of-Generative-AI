# Day 22 — Tokenization

## 🎯 Goal

Understand how raw text is converted into tokens and token IDs before it is processed by modern NLP models and LLMs.

---

## 📚 Topics Covered

- What is tokenization?
- Word tokenization
- Character tokenization
- Subword tokenization
- Byte Pair Encoding (BPE)
- WordPiece
- Unigram
- Special tokens
- Tokens vs token IDs
- Text → Tokens → IDs → Embeddings → Transformer → LLM
- Why tokenization matters for LLMs

---

## 🧠 NLP Tokenization Evolution

```text
Raw Text
   ↓
Tokenization
   ↓
Tokens
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
LLM
```

---

## 1. Word Tokenization

A sentence is divided into individual words.

Example:

```text
"I love Python"
```

becomes:

```text
["I", "love", "Python"]
```

### Problem

Vocabulary can become very large, and unknown/new words can be difficult to handle.

---

## 2. Character Tokenization

Text is divided into individual characters.

```text
"AI"
```

becomes:

```text
["A", "I"]
```

### Advantage

It can represent almost any word.

### Disadvantage

A sentence becomes very long in terms of tokens.

---

## 3. Subword Tokenization

Subword tokenization tries to find a balance between words and characters.

For example, a word could conceptually be split into smaller pieces:

```text
"unhappiness"
→ ["un", "happi", "ness"]
```

The exact tokens depend on the tokenizer vocabulary and algorithm.

Modern LLMs commonly use subword/byte-level tokenization approaches.

---

## 4. BPE — Byte Pair Encoding

BPE starts with small units and repeatedly combines frequent pairs.

Conceptually:

```text
l + o → lo
lo + v → lov
lov + e → love
```

The final vocabulary contains frequently occurring pieces.

BPE is important because it can represent both common words and previously unseen words through smaller pieces.

---

## 5. WordPiece

WordPiece is another subword tokenization approach.

It builds a vocabulary of useful subword units and represents words using those units.

It is strongly associated with Transformer models such as BERT.

---

## 6. Unigram

The Unigram approach starts with a larger vocabulary and chooses a likely segmentation of the text into subword pieces.

It is used by tokenization systems such as SentencePiece-based models.

---

## 7. Special Tokens

Tokenizers may use special tokens to communicate structure to a model.

Examples:

```text
[CLS]   → classification/start representation
[SEP]   → separator
[PAD]   → padding
[UNK]   → unknown token

<BOS>   → beginning of sequence
<EOS>   → end of sequence
```

The exact special tokens depend on the model/tokenizer.

---

## 8. Tokens vs Token IDs

A model does not directly process strings such as:

```text
["I", "love", "AI"]
```

A tokenizer maps them to numerical IDs.

Example:

```text
Tokens:
["I", "love", "AI"]

Token IDs:
[15, 82, 341]
```

The numbers are determined by the tokenizer's vocabulary.

---

## 9. Complete LLM Pipeline

```text
User Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
Predicted Token
   ↓
Next Token
   ↓
Generated Text
```

This is one of the most important pipelines to understand before learning Transformers and LLM internals.

---

## 💡 Why Tokenization Matters

Tokenization affects:

- Context-window usage
- Model input size
- Cost of API calls
- Processing speed
- How efficiently text is represented
- Handling of rare and unknown words
- Multilingual text processing

A single word is **not necessarily one token**.

For example, depending on the tokenizer:

```text
"GenerativeAI"
```

might be represented as multiple tokens.

---

## 🛠️ Implementation

Run:

```bash
python tokenization.py
```

The script demonstrates:

1. Word tokenization
2. Punctuation-aware tokenization
3. Character tokenization
4. Conceptual subword tokenization
5. Token-to-ID mapping
6. Special tokens
7. The complete text-processing pipeline

---

## 🔑 Key Takeaways

> Tokenization is the bridge between human-readable text and numerical input that language models can process.

Remember:

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

---

## 🚀 Next Step

**Day 23 — Embeddings**

After understanding tokenization, the next question is:

> How does a token ID become a meaningful numerical representation?

That leads us to **embeddings**.
