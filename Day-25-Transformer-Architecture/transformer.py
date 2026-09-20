"""Day 25: educational Transformer encoder-style implementation."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        if d_model % heads: raise ValueError("d_model must be divisible by heads")
        self.d_model, self.heads = d_model, heads
        self.head_dim = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split(self, x):
        B,T,C = x.shape
        return x.view(B,T,self.heads,self.head_dim).transpose(1,2)

    def combine(self, x):
        B,H,T,D = x.shape
        return x.transpose(1,2).contiguous().view(B,T,H*D)

    def forward(self, x, causal=False):
        Q,K,V = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        scores = Q @ K.transpose(-2,-1) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.out(self.combine(weights @ V))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model))
    def forward(self,x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.causal = causal

    def forward(self,x):
        x = x + self.drop(self.attn(self.norm1(x), self.causal))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=16, heads=4, d_ff=64, layers=2, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.position = PositionalEncoding(d_model,max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model,heads,d_ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids):
        x = self.position(self.embedding(ids))
        for block in self.blocks: x = block(x)
        return self.norm(x)

def main():
    B,T,C,H = 2,8,16,4
    ids = torch.randint(0,100,(B,T))
    model = SmallTransformer(100,C,H,64,2,T)
    out = model(ids)
    print("Token IDs:", ids.shape)
    print("Model dimension:", C)
    print("Heads:", H)
    print("Head dimension:", C//H)
    print("Transformer output:", out.shape)
    print("\nShape flow: [B,T,C] -> [B,H,T,D] -> attention -> [B,T,C]")

if __name__ == "__main__": main()
