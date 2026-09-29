"""Day 26: educational comparison of BERT, GPT and encoder-decoder architectures."""

import torch
import torch.nn as nn


class BertStyleEncoder(nn.Module):
    """BERT-style encoder-only structure."""

    def __init__(self, vocab_size=100, d_model=16, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, token_ids):
        return self.encoder(self.embedding(token_ids))


class GPTStyleDecoder(nn.Module):
    """GPT-style decoder-only causal structure."""

    def __init__(self, vocab_size=100, d_model=16, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        T = token_ids.size(1)

        # True entries above the diagonal are blocked.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=token_ids.device),
            diagonal=1
        )

        x = self.blocks(x, mask=causal_mask)
        return self.lm_head(x)


class T5StyleEncoderDecoder(nn.Module):
    """T5-style conceptual encoder-decoder structure."""

    def __init__(self, vocab_size=100, d_model=16, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, source_ids, target_ids):
        memory = self.encoder(self.embedding(source_ids))
        target = self.embedding(target_ids)
        T = target_ids.size(1)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=target_ids.device),
            diagonal=1
        )

        decoded = self.decoder(
            target, memory, tgt_mask=causal_mask
        )
        return self.lm_head(decoded)


def main():
    print("=" * 60)
    print("DAY 26 - BERT vs GPT vs ENCODER-DECODER")
    print("=" * 60)

    print("""
BERT
  Encoder-only
  Bidirectional context
  Masked Language Modeling
  Understanding

GPT
  Decoder-only
  Causal context
  Next-token prediction
  Generation

T5
  Encoder-decoder
  Cross-attention
  Text-to-text transformation
""")

    B, T, V, C, H = 2, 6, 100, 16, 4
    source = torch.randint(0, V, (B, T))
    target = torch.randint(0, V, (B, T))

    bert = BertStyleEncoder(V, C, H)
    gpt = GPTStyleDecoder(V, C, H)
    t5 = T5StyleEncoderDecoder(V, C, H)

    print("Input:", source.shape)
    print("BERT representation:", bert(source).shape)
    print("GPT logits:", gpt(source).shape)
    print("T5 logits:", t5(source, target).shape)

    print("\nMemory trick:")
    print("BERT = Understand")
    print("GPT  = Generate")
    print("T5   = Transform")


if __name__ == "__main__":
    main()
