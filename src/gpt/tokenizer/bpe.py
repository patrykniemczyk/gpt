"""Byte Pair Encoding (BPE) tokenizer implementation."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _get_pair_frequencies(token_sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """Count frequency of consecutive token pairs.
    
    Args:
        token_sequences: List of token sequences
        
    Returns:
        Dictionary mapping token pairs to their frequencies
    """
    frequencies = {}
    for tokens in token_sequences:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            frequencies[pair] = frequencies.get(pair, 0) + 1
    return frequencies


def _get_most_frequent_pair(frequencies: Dict[Tuple[int, int], int]) -> Optional[Tuple[int, int]]:
    """Get the most frequent token pair.
    
    Args:
        frequencies: Dictionary of pair frequencies
        
    Returns:
        Most frequent pair or None if no pairs exist
    """
    if not frequencies:
        return None
    return max(frequencies.items(), key=lambda x: x[1])[0]


def _merge_pair_in_sequences(
    token_sequences: List[List[int]], 
    pair: Tuple[int, int], 
    new_token_id: int
) -> List[List[int]]:
    """Replace all occurrences of a token pair with a new token.
    
    Args:
        token_sequences: List of token sequences
        pair: Token pair to replace
        new_token_id: New token ID to replace the pair with
        
    Returns:
        Updated token sequences with pairs replaced
    """
    new_sequences = []
    for tokens in token_sequences:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_sequences.append(new_tokens)
    return new_sequences


class BPETokenizer:
    """Byte Pair Encoding tokenizer with special token support.
    
    This tokenizer implements the BPE algorithm for subword tokenization,
    commonly used in modern NLP models. It supports special tokens for
    beginning of sequence (BOS), end of sequence (EOS), padding (PAD),
    and unknown tokens (UNK).
    
    Args:
        vocab_size: Target vocabulary size (excluding special tokens)
        special_tokens: Dictionary mapping special token names to strings
        
    Attributes:
        vocab_size: Target vocabulary size
        special_tokens: Special token mappings
        vocab: Mapping from token IDs to byte sequences
        merges: Mapping from token pairs to merged token IDs
        special_token_ids: Mapping from special token names to IDs
    """
    
    def __init__(
        self, 
        vocab_size: int = 2048,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "bos": "<BOS>",
            "eos": "<EOS>", 
            "pad": "<PAD>",
            "unk": "<UNK>"
        }
        
        # Initialize vocabulary with byte-level tokens (0-255)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Store merge operations: (token1, token2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # Reserve special token IDs at the end of vocabulary
        self.special_token_ids: Dict[str, int] = {}
        current_special_id = vocab_size + 256
        for name, token_str in self.special_tokens.items():
            self.special_token_ids[name] = current_special_id
            self.vocab[current_special_id] = token_str.encode('utf-8')
            current_special_id += 1
        
        logger.debug(f"Initialized BPETokenizer with vocab_size={vocab_size}")
        logger.debug(f"Special tokens: {self.special_token_ids}")

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.special_token_ids["bos"]
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID.""" 
        return self.special_token_ids["eos"]
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.special_token_ids["pad"]
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.special_token_ids["unk"]

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            min_frequency: Minimum frequency for a pair to be merged
            
        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Training texts cannot be empty")
        
        logger.info(f"Training BPE tokenizer on {len(texts)} texts")
        logger.info(f"Target vocabulary size: {self.vocab_size}")
        
        # Convert texts to byte sequences and then to token sequences
        token_sequences = []
        for text in texts:
            if text.strip():  # Skip empty texts
                text_bytes = text.encode('utf-8')
                token_sequences.append(list(text_bytes))
        
        if not token_sequences:
            raise ValueError("No valid training texts found")
        
        # Start with byte-level vocabulary (256 tokens)
        current_vocab_size = 256
        
        # Perform BPE merges until we reach target vocabulary size
        while current_vocab_size < self.vocab_size:
            # Get pair frequencies
            frequencies = _get_pair_frequencies(token_sequences)
            
            if not frequencies:
                logger.warning("No more pairs to merge")
                break
            
            # Find most frequent pair
            most_frequent_pair = _get_most_frequent_pair(frequencies)
            if most_frequent_pair is None:
                break
                
            frequency = frequencies[most_frequent_pair]
            if frequency < min_frequency:
                logger.info(f"Stopping training: max frequency {frequency} < min_frequency {min_frequency}")
                break
            
            # Create new token for this pair
            new_token_id = current_vocab_size
            
            # Store the merge operation
            self.merges[most_frequent_pair] = new_token_id
            
            # Create vocabulary entry for the new token
            token1_bytes = self.vocab[most_frequent_pair[0]]
            token2_bytes = self.vocab[most_frequent_pair[1]]
            self.vocab[new_token_id] = token1_bytes + token2_bytes
            
            # Apply the merge to all sequences
            token_sequences = _merge_pair_in_sequences(
                token_sequences, most_frequent_pair, new_token_id
            )
            
            current_vocab_size += 1
            
            if current_vocab_size % 100 == 0:
                logger.debug(f"Vocabulary size: {current_vocab_size}, "
                           f"merged pair: {most_frequent_pair} -> {new_token_id} "
                           f"(frequency: {frequency})")
        
        logger.info(f"Training complete. Final vocabulary size: {current_vocab_size}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text into token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return [self.bos_token_id, self.eos_token_id] if add_special_tokens else []
        
        # Convert to bytes and then to token sequence
        text_bytes = text.encode('utf-8')
        tokens = list(text_bytes)
        
        # Apply BPE merges in the order they were learned
        max_iterations = len(tokens) * 2  # Safety limit
        iteration_count = 0
        
        while len(tokens) > 1 and iteration_count < max_iterations:
            # Find all possible pairs in current token sequence
            pair_frequencies = _get_pair_frequencies([tokens])
            
            if not pair_frequencies:
                break
            
            # Find valid pairs (those that exist in our merge dictionary)
            valid_pairs = [pair for pair in pair_frequencies.keys() if pair in self.merges]
            
            if not valid_pairs:
                break
            
            # Choose the pair that was merged earliest (lowest merge ID)
            pair_to_merge = min(valid_pairs, key=lambda x: self.merges[x])
            
            # Apply the merge
            old_length = len(tokens)
            tokens = _merge_pair_in_sequences([tokens], pair_to_merge, self.merges[pair_to_merge])[0]
            
            # Safety check: ensure we're making progress
            if len(tokens) >= old_length:
                break
            
            iteration_count += 1
        
        # Add special tokens if requested
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""
        
        text_bytes = b""
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_token_ids.values():
                continue
            
            if token_id in self.vocab:
                text_bytes += self.vocab[token_id]
            else:
                # Handle unknown tokens
                if not skip_special_tokens:
                    text_bytes += self.vocab[self.unk_token_id]
                logger.warning(f"Unknown token ID: {token_id}")
        
        return text_bytes.decode('utf-8', errors='replace')

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save the tokenizer
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        save_data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "special_token_ids": self.special_token_ids,
            "merges": {f"{pair[0]},{pair[1]}": token_id for pair, token_id in self.merges.items()},
            "vocab": {str(token_id): vocab_bytes.decode('utf-8', errors='replace') 
                     for token_id, vocab_bytes in self.vocab.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer from file.
        
        Args:
            path: Path to load the tokenizer from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore configuration
            self.vocab_size = data["vocab_size"]
            self.special_tokens = data["special_tokens"]
            self.special_token_ids = data["special_token_ids"]
            
            # Restore merges
            self.merges = {}
            for pair_str, token_id in data["merges"].items():
                token1, token2 = map(int, pair_str.split(','))
                self.merges[(token1, token2)] = token_id
            
            # Restore vocabulary
            self.vocab = {}
            for token_id_str, vocab_str in data["vocab"].items():
                token_id = int(token_id_str)
                self.vocab[token_id] = vocab_str.encode('utf-8')
            
            logger.info(f"Tokenizer loaded from {path}")
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Error loading tokenizer: {e}")

    def get_vocab_size(self) -> int:
        """Get the full vocabulary size including special tokens.
        
        Returns:
            Total vocabulary size
        """
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into human-readable tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        token_ids = self.encode(text)
        return [self.decode([token_id], skip_special_tokens=False) for token_id in token_ids]