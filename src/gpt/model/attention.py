"""Self-attention mechanism for transformer models."""

from typing import Optional
import torch
import torch.nn as nn
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SelfAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    This implements the scaled dot-product attention mechanism used in
    transformer models, with support for causal (autoregressive) masking
    and padding mask handling.

    Args:
        embed_dim: Embedding dimension
        heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for causal mask (default: 2048)

    Raises:
        AssertionError: If embed_dim is not divisible by heads
    """

    def __init__(
        self, embed_dim: int, heads: int, dropout: float = 0.1, max_seq_len: int = 2048
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (
            self.head_dim * heads == embed_dim
        ), f"embed_dim ({embed_dim}) must be divisible by heads ({heads})"

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as buffer (not a parameter)
        causal_mask = (
            torch.tril(
                torch.ones(
                    max_seq_len,
                    max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer("causal_mask", causal_mask)

        logger.debug(
            f"Initialized SelfAttention: embed_dim={embed_dim}, heads={heads}, "
            f"head_dim={self.head_dim}, dropout={dropout}"
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Boolean mask of shape (batch_size, seq_len) where
                         True indicates padding tokens to ignore

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, seq_len, heads, head_dim) -> (batch_size, heads,
        # seq_len, head_dim)
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        # Shape: (batch_size, heads, seq_len, seq_len)
        attention_scores = torch.einsum(
            "bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len].to(x.device)
        attention_scores = attention_scores.masked_fill(
            causal_mask == 0, float("-inf"))

        # Apply padding mask if provided
        if padding_mask is not None:
            # Expand padding mask for all heads: (batch_size, 1, 1, seq_len)
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                padding_mask_expanded, float("-inf")
            )

        # Compute attention weights and apply dropout
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # Shape: (batch_size, heads, seq_len, head_dim)
        attention_output = torch.einsum(
            "bhij,bhjd->bhid", attention_weights, v)

        # Concatenate heads and project to output dimension
        # Shape: (batch_size, seq_len, embed_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # Final output projection
        output = self.out_proj(attention_output)
        output = self.dropout(output)

        # Zero out outputs corresponding to padding tokens
        if padding_mask is not None:
            # Shape: (batch_size, seq_len, 1)
            non_padding_mask = (~padding_mask).unsqueeze(-1).type_as(output)
            output = output * non_padding_mask

        return output
