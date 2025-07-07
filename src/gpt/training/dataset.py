"""Dataset utilities for text data loading and preprocessing."""

import json
import torch
from pathlib import Path
from typing import List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from ..tokenizer import BPETokenizer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextDataset(Dataset):
    """Dataset for text sequences with proper padding and special tokens.
    
    This dataset handles tokenized text sequences with automatic padding,
    special token addition, and proper attention mask generation for
    efficient batching during training.
    
    Args:
        tokenized_texts: List of tokenized text sequences
        max_length: Maximum sequence length (will pad/truncate to this length)
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID  
        pad_token_id: Padding token ID
    """
    
    def __init__(
        self,
        tokenized_texts: List[List[int]],
        max_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int
    ) -> None:
        self.tokenized_texts = tokenized_texts
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        logger.info(f"Created TextDataset with {len(tokenized_texts)} samples, "
                   f"max_length={max_length}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_ids, target_ids) tensors
        """
        tokens = self.tokenized_texts[idx]
        
        # Truncate if too long (leave space for BOS/EOS)
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # Create input sequence: [BOS] + tokens
        input_tokens = [self.bos_token_id] + tokens
        
        # Create target sequence: tokens + [EOS]
        target_tokens = tokens + [self.eos_token_id]
        
        # Pad sequences to max_length
        while len(input_tokens) < self.max_length:
            input_tokens.append(self.pad_token_id)
        while len(target_tokens) < self.max_length:
            target_tokens.append(self.pad_token_id)
        
        # Ensure exact length
        input_tokens = input_tokens[:self.max_length]
        target_tokens = target_tokens[:self.max_length]
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )


def load_text_data(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_file: Optional[Union[str, Path]] = None,
    text_column: str = "text"
) -> List[str]:
    """Load text data from HuggingFace datasets or cache.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        dataset_config: Optional dataset configuration
        split: Dataset split to load
        num_samples: Maximum number of samples to load
        cache_file: Optional cache file to save/load data
        text_column: Name of the text column in the dataset
        
    Returns:
        List of text strings
        
    Raises:
        ValueError: If no valid texts are found
    """
    # Try to load from cache first
    if cache_file and Path(cache_file).exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            logger.info(f"Loaded {len(texts)} texts from cache")
            return texts
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to load cache file: {e}, loading from source")
    
    # Load from HuggingFace datasets
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        logger.info(f"Dataset config: {dataset_config}")
    
    try:
        # Use streaming for large datasets
        streaming = num_samples is not None and num_samples > 50000
        dataset = load_dataset(
            dataset_name, 
            dataset_config, 
            split=split, 
            streaming=streaming
        )
        
        texts = []
        for i, sample in enumerate(dataset):
            if text_column not in sample:
                logger.warning(f"Text column '{text_column}' not found in sample {i}")
                continue
                
            text = sample[text_column]
            if text and text.strip():  # Only include non-empty texts
                texts.append(text.strip())
            
            if num_samples and len(texts) >= num_samples:
                break
        
        if not texts:
            raise ValueError(f"No valid texts found in dataset {dataset_name}")
        
        logger.info(f"Loaded {len(texts)} texts from dataset")
        
        # Save to cache if specified
        if cache_file:
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached dataset to {cache_file}")
        
        return texts
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def prepare_datasets(
    texts: List[str],
    tokenizer: BPETokenizer,
    max_length: int,
    validation_split: float = 0.1,
    train_shuffle: bool = True
) -> Tuple[List[List[int]], List[List[int]]]:
    """Prepare training and validation datasets.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        validation_split: Fraction of data to use for validation
        train_shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_tokenized, val_tokenized) lists
        
    Raises:
        ValueError: If validation_split is invalid
    """
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1 (exclusive)")
    
    logger.info(f"Tokenizing {len(texts)} texts...")
    
    # Tokenize all texts
    tokenized_texts = []
    for i, text in enumerate(texts):
        try:
            tokens = tokenizer.encode(text)
            if tokens:  # Only include non-empty tokenized texts
                tokenized_texts.append(tokens)
        except Exception as e:
            logger.warning(f"Failed to tokenize text {i}: {e}")
    
    if not tokenized_texts:
        raise ValueError("No texts could be tokenized")
    
    logger.info(f"Successfully tokenized {len(tokenized_texts)} texts")
    
    # Split into train and validation
    if validation_split > 0:
        split_idx = int(len(tokenized_texts) * (1 - validation_split))
        train_tokenized = tokenized_texts[:split_idx]
        val_tokenized = tokenized_texts[split_idx:]
        
        if train_shuffle:
            import random
            random.shuffle(train_tokenized)
        
        logger.info(f"Split into {len(train_tokenized)} train and "
                   f"{len(val_tokenized)} validation samples")
    else:
        train_tokenized = tokenized_texts
        val_tokenized = []
        
        if train_shuffle:
            import random
            random.shuffle(train_tokenized)
        
        logger.info(f"Using all {len(train_tokenized)} samples for training")
    
    return train_tokenized, val_tokenized


def create_dataloader(
    tokenized_texts: List[List[int]],
    tokenizer: BPETokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for tokenized texts.
    
    Args:
        tokenized_texts: List of tokenized text sequences
        tokenizer: Tokenizer instance (for special token IDs)
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = TextDataset(
        tokenized_texts=tokenized_texts,
        max_length=max_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    def collate_fn(batch):
        """Custom collate function to stack tensors."""
        input_ids, target_ids = zip(*batch)
        return torch.stack(input_ids), torch.stack(target_ids)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )