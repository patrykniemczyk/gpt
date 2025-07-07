"""
GPT: A minimal from-scratch implementation of GPT architecture.

This package provides a clean, well-documented implementation of the GPT
(Generative Pre-trained Transformer) architecture with modern Python
best practices.
"""

__version__ = "0.1.0"
__author__ = "GPT Implementation"

from .model import GPT, SelfAttention, FeedForward, TransformerBlock
from .tokenizer import BPETokenizer
from .config import (
    GPTConfig, ModelConfig, TrainingConfig, DataConfig,
    TokenizerConfig, SamplingConfig, FilesConfig, 
    load_config, save_config
)

__all__ = [
    "GPT",
    "SelfAttention", 
    "FeedForward",
    "TransformerBlock",
    "BPETokenizer",
    "GPTConfig",
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "TokenizerConfig",
    "SamplingConfig",
    "FilesConfig",
    "load_config",
    "save_config",
]