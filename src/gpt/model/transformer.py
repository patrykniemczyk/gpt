"""Transformer block components."""

from typing import Optional
import torch
import torch.nn as nn
from .attention import SelfAttention
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    This implements the feed-forward network used in transformer blocks,
    consisting of two linear transformations with a GELU activation in between.

    Args:
        embed_dim: Input/output embedding dimension
        ff_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, embed_dim: int, ff_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        logger.debug(
            f"Initialized FeedForward: embed_dim={embed_dim}, ff_dim={ff_dim}, "
            f"dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward.

    This implements a standard transformer block consisting of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Position-wise feed-forward network with residual connection and layer norm

    The layer norm is applied before the sub-layers (pre-norm), which is
    commonly used in modern transformer implementations.

    Args:
        embed_dim: Embedding dimension
        heads: Number of attention heads
        ff_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for attention (default: 2048)
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.ff_dim = ff_dim

        # Self-attention mechanism
        self.attention = SelfAttention(
            embed_dim=embed_dim, heads=heads, dropout=dropout, max_seq_len=max_seq_len
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embed_dim=embed_dim, ff_dim=ff_dim, dropout=dropout
        )

        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        logger.debug(
            f"Initialized TransformerBlock: embed_dim={embed_dim}, heads={heads}, "
            f"ff_dim={ff_dim}, dropout={dropout}"
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Boolean mask of shape (batch_size, seq_len) where
                         True indicates padding tokens to ignore

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection (pre-norm)
        attention_output = self.attention(self.norm1(x), padding_mask)
        x = x + attention_output

        # Feed-forward with residual connection (pre-norm)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x
