"""Main GPT model implementation."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from .transformer import TransformerBlock
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GPT(nn.Module):
    """GPT (Generative Pre-trained Transformer) model.
    
    This implements the core GPT architecture consisting of:
    1. Token and positional embeddings
    2. Stack of transformer blocks
    3. Final layer normalization
    4. Language modeling head
    
    The model supports causal (autoregressive) generation with proper
    handling of padding tokens and attention masking.
    
    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        ff_dim: Hidden dimension of feed-forward networks
        num_layers: Number of transformer layers
        heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 2048)
        pad_token_id: ID of padding token (default: 0)
        
    Raises:
        AssertionError: If embed_dim is not divisible by heads
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        ff_dim: int,
        num_layers: int,
        heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_token_id: int = 0
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                heads=heads,
                ff_dim=ff_dim,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # Register position indices as buffer
        self.register_buffer(
            "position_indices",
            torch.arange(max_seq_len).unsqueeze(0)
        )

        # Initialize weights
        self.apply(self._init_weights)
        
        # Log model configuration
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Initialized GPT model:")
        logger.info(f"  - vocab_size: {vocab_size}")
        logger.info(f"  - embed_dim: {embed_dim}")
        logger.info(f"  - ff_dim: {ff_dim}")
        logger.info(f"  - num_layers: {num_layers}")
        logger.info(f"  - heads: {heads}")
        logger.info(f"  - max_seq_len: {max_seq_len}")
        logger.info(f"  - total_params: {total_params:,}")
        logger.info(f"  - trainable_params: {trainable_params:,}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT conventions.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create padding mask from input token IDs.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Boolean mask of shape (batch_size, seq_len) where True indicates padding
        """
        return input_ids == self.pad_token_id

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of GPT model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask. If None, will be created
                           from padding tokens automatically.
                           
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)

        # Get position indices for this sequence length
        position_ids = self.position_indices[:, :seq_len].expand(batch_size, seq_len)
        position_ids = position_ids.to(input_ids.device)

        # Compute embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep only top p probability mass for sampling
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            eos_token_id: Token ID to stop generation
            
        Returns:
            Generated token IDs including input
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if sequence is too long
                if input_ids.size(1) >= self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]

                # Forward pass
                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

                # Sample next token
                if do_sample:
                    # Handle case where all logits are -inf
                    if torch.all(torch.isinf(next_token_logits) & (next_token_logits < 0)):
                        next_token_logits = torch.zeros_like(next_token_logits)
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for end of sequence
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def save_pretrained(self, save_directory: str) -> None:
        """Save model state and configuration.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save model configuration
        config = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "heads": self.heads,
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: str) -> "GPT":
        """Load model from saved state and configuration.
        
        Args:
            model_directory: Directory containing saved model
            
        Returns:
            Loaded GPT model
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        weights_path = os.path.join(model_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        logger.info(f"Model loaded from {model_directory}")
        return model