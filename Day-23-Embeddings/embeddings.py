"""
Day 23 - Embeddings

Learning implementation:
- Token IDs -> embedding vectors
- Embedding dimensions
- Cosine similarity
- Semantic similarity
- Conceptual RAG workflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def show_embedding_layer():
    vocabulary_size = 10
    embedding_dimension = 4

    embedding = nn.Embedding(
        num_embeddings=vocabulary_size,
        embedding_dim=embedding_dimension
    )

    token_ids = torch.tensor([1, 3, 5, 7])
    vectors = embedding(token_ids)

    print("Token IDs:")
    print(token_ids)

    print("\nEmbedding vectors:")
    print(vectors)

    print("\nShape:")
    print(vectors.shape)

    print("\nShape means: (number of tokens, embedding dimension)")


def cosine_similarity_demo():
    vector_a = torch.tensor([[1.0, 0.0, 1.0]])
    vector_b = torch.tensor([[1.0, 0.0, 0.9]])
    vector_c = torch.tensor([[0.0, 1.0, 0.0]])

    similarity_ab = F.cosine_similarity(vector_a, vector_b).item()
    similarity_ac = F.cosine_similarity(vector_a, vector_c).item()

    print("Similarity(A, B):", round(similarity_ab, 4))
    print("Similarity(A, C):", round(similarity_ac, 4))


def semantic_similarity_demo():
    # Toy vectors for learning only; these are not trained embeddings.
    embeddings = {
        "cat": torch.tensor([[0.90, 0.80, 0.70]]),
        "kitten": torch.tensor([[0.88, 0.79, 0.72]]),
        "car": torch.tensor([[0.10, 0.30, 0.95]]),
    }

    cat_kitten = F.cosine_similarity(
        embeddings["cat"], embeddings["kitten"]
    ).item()

    cat_car = F.cosine_similarity(
        embeddings["cat"], embeddings["car"]
    ).item()

    print("Similarity(cat, kitten):", round(cat_kitten, 4))
    print("Similarity(cat, car):", round(cat_car, 4))


def main():
    print("=" * 60)
    print("DAY 23 - EMBEDDINGS")
    print("=" * 60)

    print("\n1. TOKEN IDs -> EMBEDDINGS")
    print("-" * 60)
    show_embedding_layer()

    print("\n2. COSINE SIMILARITY")
    print("-" * 60)
    cosine_similarity_demo()

    print("\n3. SEMANTIC SIMILARITY")
    print("-" * 60)
    semantic_similarity_demo()

    print("\n4. EMBEDDINGS IN RAG")
    print("-" * 60)
    print("""
Documents
    ↓
Chunking
    ↓
Embedding Model
    ↓
Vectors
    ↓
Vector Database

User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
LLM
    ↓
Answer
""")

    print("=" * 60)
    print("DAY 23 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
