"""Day 21 - NLP Evolution"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# --------------------------------------------------
# Documents
# --------------------------------------------------

documents = [
    "I love Python",
    "I love AI",
    "Python is powerful",
]


# --------------------------------------------------
# 1. BAG OF WORDS
# --------------------------------------------------

print("=" * 60)
print("1. BAG OF WORDS")
print("=" * 60)

bow = CountVectorizer()

bow_matrix = bow.fit_transform(documents)

print("Vocabulary:")
print(bow.get_feature_names_out())

print("\nMatrix:")
print(bow_matrix.toarray())


# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

print("\n" + "=" * 60)
print("2. TF-IDF")
print("=" * 60)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(documents)

print("Vocabulary:")
print(tfidf.get_feature_names_out())

print("\nMatrix:")
print(tfidf_matrix.toarray())


# --------------------------------------------------
# 3. WORD2VEC
# --------------------------------------------------

print("\n" + "=" * 60)
print("3. WORD2VEC")
print("=" * 60)

print(
    "Words can be represented as dense vectors "
    "called embeddings."
)


# --------------------------------------------------
# 4. RNN
# --------------------------------------------------

print("\n" + "=" * 60)
print("4. RNN")
print("=" * 60)

print(
    "RNN processes sequences and carries information "
    "through hidden states."
)


# --------------------------------------------------
# 5. LSTM
# --------------------------------------------------

print("\n" + "=" * 60)
print("5. LSTM")
print("=" * 60)

print(
    "LSTM improves sequence memory using gates."
)


# --------------------------------------------------
# 6. ATTENTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("6. ATTENTION")
print("=" * 60)

print(
    "Attention allows a model to focus on relevant "
    "parts of the input."
)


# --------------------------------------------------
# 7. TRANSFORMER
# --------------------------------------------------

print("\n" + "=" * 60)
print("7. TRANSFORMER")
print("=" * 60)

print(
    "Transformers use attention as the central mechanism."
)


# --------------------------------------------------
# 8. LLM
# --------------------------------------------------

print("\n" + "=" * 60)
print("8. LARGE LANGUAGE MODEL")
print("=" * 60)

print(
    "Large Language Models are large neural networks "
    "trained on huge text datasets."
)


# --------------------------------------------------
# COMPLETE NLP EVOLUTION
# --------------------------------------------------

print("\n" + "=" * 60)
print("NLP EVOLUTION")
print("=" * 60)

evolution = [
    "Bag of Words",
    "TF-IDF",
    "Word2Vec",
    "RNN",
    "LSTM",
    "Attention",
    "Transformer",
    "LLM",
]

for i, item in enumerate(evolution, start=1):
    print(f"{i}. {item}")