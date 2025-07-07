#!/usr/bin/env python3
"""Demo script showing the complete GPT training and inference workflow."""

import sys
from pathlib import Path
import torch
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt.config import GPTConfig
from gpt.model import GPT
from gpt.tokenizer import BPETokenizer
from gpt.training import Trainer
from gpt.training.dataset import prepare_datasets, create_dataloader
from gpt.utils.logging import setup_logging, get_logger


def create_demo_data():
    """Create some demo training data."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence for training.",
        "Machine learning and artificial intelligence are fascinating.",
        "Natural language processing with transformers is powerful.",
        "Deep learning models can generate human-like text.",
        "Python is a great programming language for AI.",
        "The future of technology looks very promising.",
        "Large language models are changing the world.",
        "Training neural networks requires careful tuning.",
        "Text generation has many practical applications."
    ] * 10  # Repeat to have more training data


def main():
    """Run the complete demo."""
    print("🚀 GPT Implementation Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="INFO", console_output=True)
    logger = get_logger(__name__)
    
    # Create demo configuration
    config = GPTConfig()
    config.model.embed_dim = 128
    config.model.ff_dim = 512
    config.model.num_layers = 2
    config.model.heads = 8
    config.model.max_block_size = 64
    config.model.vocab_size = 512
    config.training.batch_size = 4
    config.training.epochs = 2
    config.training.learning_rate = 1e-3
    config.data.validation_split = 0.2
    
    device = torch.device("cpu")  # Use CPU for demo
    
    print("\n📊 Configuration:")
    print(f"  Model: {config.model.embed_dim}d, {config.model.num_layers} layers")
    print(f"  Vocab: {config.model.vocab_size} tokens")
    print(f"  Training: {config.training.epochs} epochs, batch_size={config.training.batch_size}")
    
    # Create demo data
    print("\n📚 Creating demo training data...")
    texts = create_demo_data()
    print(f"  Created {len(texts)} training samples")
    
    # Initialize tokenizer
    print("\n🔤 Training tokenizer...")
    tokenizer = BPETokenizer(
        vocab_size=config.model.vocab_size - len(config.tokenizer.special_tokens),
        special_tokens=config.tokenizer.special_tokens
    )
    tokenizer.train(texts[:50])  # Train on subset
    print(f"  Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test tokenization
    test_text = "Hello world, this is a test."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
    
    # Prepare datasets
    print("\n📦 Preparing datasets...")
    train_tokenized, val_tokenized = prepare_datasets(
        texts=texts,
        tokenizer=tokenizer,
        max_length=config.model.max_block_size,
        validation_split=config.data.validation_split
    )
    
    train_dataloader = create_dataloader(
        tokenized_texts=train_tokenized,
        tokenizer=tokenizer,
        max_length=config.model.max_block_size,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        tokenized_texts=val_tokenized,
        tokenizer=tokenizer,
        max_length=config.model.max_block_size,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(val_dataloader)}")
    
    # Initialize model
    print("\n🧠 Creating model...")
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
    
    print(f"  Model parameters: {model.get_num_params():,}")
    
    # Test generation before training
    print("\n🎲 Generation before training:")
    sample_before = model.generate(
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        max_new_tokens=20,
        temperature=1.0,
        do_sample=True
    )
    text_before = tokenizer.decode(sample_before[0].tolist(), skip_special_tokens=True)
    print(f"  Generated: '{text_before}'")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    config.files.output_dir = temp_dir
    config.files.log_dir = temp_dir + "/logs"
    
    try:
        # Initialize trainer
        print("\n🏋️ Training model...")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
        
        # Train the model
        results = trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            resume_from_checkpoint=False
        )
        
        print(f"\n✅ Training completed!")
        print(f"  Final results: {results}")
        
        # Test generation after training
        print("\n🎯 Generation after training:")
        sample_after = trainer.generate_sample(
            prompt="Hello world",
            max_new_tokens=30,
            temperature=0.8
        )
        print(f"  Generated: '{sample_after}'")
        
        # Test model save/load
        print("\n💾 Testing model save/load...")
        model_dir = Path(temp_dir) / "saved_model"
        model.save_pretrained(str(model_dir))
        
        loaded_model = GPT.from_pretrained(str(model_dir))
        print(f"  Model saved and loaded successfully")
        
        # Final generation test
        print("\n🔮 Final generation test:")
        with torch.no_grad():
            final_sample = loaded_model.generate(
                torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
                max_new_tokens=25,
                temperature=0.7
            )
        final_text = tokenizer.decode(final_sample[0].tolist(), skip_special_tokens=True)
        print(f"  Final generated text: '{final_text}'")
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 50)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()