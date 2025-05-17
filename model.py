import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.hd = embed_dim // heads

        assert self.hd * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # [B, T, H, D]
        queries = self.queries(x).view(
            batch_size, seq_len, self.heads, self.hd)
        keys = self.keys(x).view(batch_size, seq_len, self.heads, self.hd)
        values = self.values(x).view(batch_size, seq_len, self.heads, self.hd)

        # [B, H, T, D]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores: [B, H, T, T]
        energy = torch.einsum(
            "bhid,bhjd->bhij", [queries, keys]) / (self.hd ** 0.5)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        energy = energy.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # [B, H, T, D]
        out = torch.einsum("bhij,bhjd->bhid", attention, values)

        # Reshape and final linear: [B, T, H*D]
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        out = self.linear(out)
        out = self.dropout(out)
        return out
