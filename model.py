import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1, max_seq_len=2048):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.hd = embed_dim // heads
        assert self.hd * heads == embed_dim, "embed_dim must be divisible by heads"

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)
                          ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x, padding_mask=None):
        B, T, _ = x.size()

        q = self.q(x).view(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.hd).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q,
                              k) / (self.hd ** 0.5)

        causal_mask = self.mask[:, :, :T, :T].to(x.device)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if padding_mask is not None:
            pad_k = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_k, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        out = self.linear(out)
        out = self.dropout(out)

        if padding_mask is not None:
            nonpad = (~padding_mask).unsqueeze(-1).type_as(out)
            out = out * nonpad

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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1, max_seq_len=2048):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, heads, dropout, max_seq_len)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        x = x + self.attention(self.norm1(x), padding_mask)
        x = x + self.feed_forward(self.norm2(x))

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        ff_dim,
        num_layers,
        heads,
        dropout=0.1,
        max_seq_len=2048,
        pad_token_id=0
    ):
        super(GPT, self).__init__()

        self.pad_token_id = pad_token_id
        self.embed_dim = embed_dim

        self.te = nn.Embedding(vocab_size, embed_dim)
        self.pe = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, heads, ff_dim,
                             dropout=dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.register_buffer(
            "positions_buffer",
            torch.arange(max_seq_len).unsqueeze(0)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def create_padding_mask(self, x):
        return x == self.pad_token_id

    def forward(self, x):
        B, T = x.size()

        padding_mask = self.create_padding_mask(x)
        positions = self.positions_buffer[:, :T].expand(B, T).to(x.device)

        x = self.te(x) + self.pe(positions)

        for layer in self.layers:
            x = layer(x, padding_mask)

        x = self.ln_f(x)

        logits = self.head(x)
        return logits
