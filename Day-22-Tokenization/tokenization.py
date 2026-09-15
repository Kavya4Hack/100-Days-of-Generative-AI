"""
Day 22 - Tokenization

Beginner-friendly demonstration of:
1. Word tokenization
2. Punctuation-aware tokenization
3. Character tokenization
4. Conceptual subword tokenization
5. Token IDs
6. Special tokens
7. Complete text -> tokens -> IDs pipeline

This is a learning implementation.
Production LLMs use model-specific tokenizers such as
BPE/byte-level BPE, WordPiece, Unigram, or related methods.
"""

import re


def word_tokenization(text):
    """Simple word tokenization using whitespace."""
    return text.split()


def punctuation_aware_tokenization(text):
    """Keep words and punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", text)


def character_tokenization(text):
    """Split text into individual characters."""
    return list(text)


def conceptual_subword_tokenization():
    """
    A conceptual example only.

    Real subword tokenizers learn their vocabulary and
    segmentation rules from training data.
    """
    word = "unhappiness"
    tokens = ["un", "happi", "ness"]

    return word, tokens


def token_to_id(tokens):
    """
    Small educational vocabulary.

    Real tokenizers have much larger vocabularies.
    """
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "I": 2,
        "love": 3,
        "Python": 4,
        "AI": 5,
        ".": 6,
    }

    ids = []

    for token in tokens:
        ids.append(vocabulary.get(token, vocabulary["[UNK]"]))

    return ids, vocabulary


def main():
    text = "I love Python."

    print("=" * 60)
    print("DAY 22 - TOKENIZATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. WORD TOKENIZATION
    # ---------------------------------------------------------
    print("\n1. WORD TOKENIZATION")
    print("-" * 60)

    words = word_tokenization(text)

    print("Text:")
    print(text)

    print("\nTokens:")
    print(words)

    # ---------------------------------------------------------
    # 2. PUNCTUATION-AWARE TOKENIZATION
    # ---------------------------------------------------------
    print("\n2. PUNCTUATION-AWARE TOKENIZATION")
    print("-" * 60)

    tokens_with_punctuation = punctuation_aware_tokenization(text)

    print("Tokens:")
    print(tokens_with_punctuation)

    # ---------------------------------------------------------
    # 3. CHARACTER TOKENIZATION
    # ---------------------------------------------------------
    print("\n3. CHARACTER TOKENIZATION")
    print("-" * 60)

    characters = character_tokenization("AI")

    print("Text: AI")
    print("Characters:")
    print(characters)

    # ---------------------------------------------------------
    # 4. CONCEPTUAL SUBWORD TOKENIZATION
    # ---------------------------------------------------------
    print("\n4. SUBWORD TOKENIZATION")
    print("-" * 60)

    word, subwords = conceptual_subword_tokenization()

    print(f"Word: {word}")
    print(f"Conceptual subword tokens: {subwords}")

    # ---------------------------------------------------------
    # 5. TOKENS -> TOKEN IDS
    # ---------------------------------------------------------
    print("\n5. TOKENS -> TOKEN IDS")
    print("-" * 60)

    tokens = ["I", "love", "AI", "."]

    ids, vocabulary = token_to_id(tokens)

    print("Vocabulary:")
    print(vocabulary)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(ids)

    # ---------------------------------------------------------
    # 6. SPECIAL TOKENS
    # ---------------------------------------------------------
    print("\n6. SPECIAL TOKENS")
    print("-" * 60)

    special_tokens = [
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[UNK]",
        "<BOS>",
        "<EOS>",
    ]

    for token in special_tokens:
        print(token)

    # ---------------------------------------------------------
    # 7. COMPLETE PIPELINE
    # ---------------------------------------------------------
    print("\n7. COMPLETE TEXT PROCESSING PIPELINE")
    print("-" * 60)

    sentence = "I love AI."

    tokens = punctuation_aware_tokenization(sentence)

    # Our small educational vocabulary does not contain
    # every punctuation-aware token, so unknown tokens
    # become [UNK].
    ids, _ = token_to_id(tokens)

    print("Text:")
    print(sentence)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(ids)

    print("\nConceptual next steps:")
    print("Token IDs")
    print("   ↓")
    print("Embeddings")
    print("   ↓")
    print("Transformer")
    print("   ↓")
    print("LLM output")

    print("\n" + "=" * 60)
    print("DAY 22 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
