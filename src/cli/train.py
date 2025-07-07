"""Command line interface for training GPT models."""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt.config import load_config, GPTConfig
from gpt.model import GPT
from gpt.tokenizer import BPETokenizer
from gpt.training import Trainer
from gpt.training.dataset import load_text_data, prepare_datasets, create_dataloader
from gpt.utils.logging import setup_logging, get_logger


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train a GPT model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--tokenizer-only", 
        action="store_true",
        help="Only train the tokenizer, don't train the model"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run through setup without actual training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        return 1
    
    # Setup logging
    setup_logging(log_dir=config.files.log_dir, log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting GPT training")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load training data
        logger.info("Loading training data...")
        texts = load_text_data(
            dataset_name=config.data.dataset_name,
            dataset_config=config.data.dataset_config,
            split=config.data.dataset_split,
            num_samples=config.data.num_training_samples,
            cache_file=config.data.data_file
        )
        logger.info(f"Loaded {len(texts)} texts")
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = BPETokenizer(
            vocab_size=config.tokenizer.vocab_size or config.model.vocab_size - len(config.tokenizer.special_tokens),
            special_tokens=config.tokenizer.special_tokens
        )
        
        # Train or load tokenizer
        tokenizer_path = Path(config.tokenizer.path)
        if tokenizer_path.exists():
            logger.info(f"Loading existing tokenizer from {tokenizer_path}")
            tokenizer.load(tokenizer_path)
        else:
            logger.info("Training tokenizer...")
            tokenizer_texts = texts[:config.data.tokenizer_training_samples]
            tokenizer.train(tokenizer_texts)
            logger.info(f"Saving tokenizer to {tokenizer_path}")
            tokenizer.save(tokenizer_path)
        
        logger.info(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
        
        # If tokenizer-only mode, exit here
        if args.tokenizer_only:
            logger.info("Tokenizer-only mode: training complete")
            return 0
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_tokenized, val_tokenized = prepare_datasets(
            texts=texts,
            tokenizer=tokenizer,
            max_length=config.model.max_block_size,
            validation_split=config.data.validation_split
        )
        
        # Create data loaders
        train_dataloader = create_dataloader(
            tokenized_texts=train_tokenized,
            tokenizer=tokenizer,
            max_length=config.model.max_block_size,
            batch_size=config.training.batch_size,
            shuffle=True
        )
        
        eval_dataloader = None
        if val_tokenized:
            eval_dataloader = create_dataloader(
                tokenized_texts=val_tokenized,
                tokenizer=tokenizer,
                max_length=config.model.max_block_size,
                batch_size=config.training.batch_size,
                shuffle=False
            )
        
        # Initialize model
        logger.info("Initializing model...")
        model = GPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=config.model.embed_dim,
            ff_dim=config.model.ff_dim,
            num_layers=config.model.num_layers,
            heads=config.model.heads,
            dropout=config.model.dropout,
            max_seq_len=config.model.max_block_size,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run complete - setup successful")
            return 0
        
        # Generate initial sample
        logger.info("Generating initial sample...")
        initial_sample = trainer.generate_sample(
            prompt="The future of artificial intelligence",
            max_new_tokens=50,
            temperature=config.sampling.temperature_default
        )
        logger.info(f"Initial sample: {initial_sample}")
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            resume_from_checkpoint=args.resume
        )
        
        # Generate final sample
        logger.info("Generating final sample...")
        final_sample = trainer.generate_sample(
            prompt="The future of artificial intelligence",
            max_new_tokens=100,
            temperature=config.sampling.temperature_default
        )
        logger.info(f"Final sample: {final_sample}")
        
        # Log training results
        logger.info("Training completed successfully!")
        logger.info(f"Training results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        return 1


if __name__ == "__main__":
    sys.exit(main())