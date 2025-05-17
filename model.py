import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, max_seq_len=2048):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.hd = embed_dim // heads
        assert self.hd * heads == embed_dim, "embed_dim must be divisible by heads"

        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Precompute causal mask [1, 1, T, T] (broadcastable)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)
                          ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections and reshape
        queries = self.queries(x).view(
            B, T, self.heads, self.hd).transpose(1, 2)
        keys = self.keys(x).view(
            B, T, self.heads, self.hd).transpose(1, 2)
        values = self.values(x).view(
            B, T, self.heads, self.hd).transpose(1, 2)

        # Attention scores
        scores = torch.einsum(
            "bhid,bhjd->bhij", queries, keys) / (self.hd ** 0.5)

        # Apply cached causal mask (trimmed to current T)
        causal_mask = self.mask[:, :, :T, :T].to(x.device)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Attention weights
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = torch.einsum("bhij,bhjd->bhid", attn, values)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.linear(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)
