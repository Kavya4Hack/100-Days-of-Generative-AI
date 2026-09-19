"""
Day 24 - Multi-Head Attention

Learning implementation using PyTorch:
- scaled dot-product attention
- attention weights
- causal masking
- multi-head self-attention
- tensor shapes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute softmax(QK^T / sqrt(d_k))V."""

    d_k = Q.size(-1)

    # Similarity between every query and key.
    scores = Q @ K.transpose(-2, -1)

    # Scale the scores.
    scores = scores / math.sqrt(d_k)

    # Optional causal/padding mask.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Convert scores into attention weights.
    weights = F.softmax(scores, dim=-1)

    # Combine Value vectors using the weights.
    output = weights @ V

    return output, weights


def make_causal_mask(sequence_length):
    """Lower-triangular mask: current and previous positions are visible."""

    return torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )


class MultiHeadAttention(nn.Module):
    """Small educational implementation of multi-head self-attention."""

    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """[B,T,C] -> [B,H,T,D]."""

        B, T, _ = x.shape

        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """[B,H,T,D] -> [B,T,C]."""

        B, H, T, D = x.shape

        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * D)

    def forward(self, x, causal=True):

        Q = self.split_heads(self.q_projection(x))
        K = self.split_heads(self.k_projection(x))
        V = self.split_heads(self.v_projection(x))

        mask = None

        if causal:
            mask = make_causal_mask(x.size(1)).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)

        attended, weights = scaled_dot_product_attention(
            Q, K, V, mask
        )

        combined = self.combine_heads(attended)

        output = self.output_projection(combined)

        return output, weights


def basic_attention_demo():
    print("\n1. SCALED DOT-PRODUCT ATTENTION")
    print("-" * 60)

    # B=1, H=1, T=3, D=4
    Q = torch.tensor(
        [[
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]]
    ).unsqueeze(1)

    K = Q.clone()

    V = torch.tensor(
        [[
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]]
    ).unsqueeze(1)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("Weights shape:", weights.shape)
    print("Output shape:", output.shape)

    print("\nAttention weights:")
    print(weights)


def causal_mask_demo():
    print("\n2. CAUSAL MASK")
    print("-" * 60)

    mask = make_causal_mask(5)

    print(mask.int())

    print("\n1 = visible")
    print("0 = blocked future position")


def multi_head_demo():
    print("\n3. MULTI-HEAD SELF-ATTENTION")
    print("-" * 60)

    B = 2
    T = 5
    C = 8
    H = 2

    x = torch.randn(B, T, C)

    attention = MultiHeadAttention(
        d_model=C,
        num_heads=H
    )

    output, weights = attention(x, causal=True)

    print("Input:", x.shape)
    print("Heads:", H)
    print("Head dimension:", attention.head_dim)
    print("Weights:", weights.shape)
    print("Output:", output.shape)

    print("\nShape flow:")
    print("[B,T,C] -> [B,H,T,D] -> [B,H,T,T] -> [B,T,C]")


def main():
    print("=" * 60)
    print("DAY 24 - MULTI-HEAD ATTENTION")
    print("=" * 60)

    basic_attention_demo()
    causal_mask_demo()
    multi_head_demo()

    print("\n" + "=" * 60)
    print("DAY 24 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
