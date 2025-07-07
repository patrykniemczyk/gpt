"""Unit tests for BPE tokenizer."""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt.tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    """Test BPE tokenizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.special_tokens = {
            "bos": "<BOS>",
            "eos": "<EOS>",
            "pad": "<PAD>",
            "unk": "<UNK>"
        }
        self.sample_texts = [
            "Hello world! How are you today?",
            "This is a test sentence for the tokenizer.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful.",
            "Natural language processing with transformers."
        ]
    
    def test_tokenizer_creation(self):
        """Test tokenizer initialization."""
        tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        
        self.assertEqual(tokenizer.vocab_size, self.vocab_size)
        self.assertEqual(tokenizer.special_tokens, self.special_tokens)
        self.assertEqual(len(tokenizer.vocab), 256 + len(self.special_tokens))  # Byte level + special tokens
    
    def test_special_token_ids(self):
        """Test special token ID properties."""
        tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        
        self.assertIsInstance(tokenizer.bos_token_id, int)
        self.assertIsInstance(tokenizer.eos_token_id, int)
        self.assertIsInstance(tokenizer.pad_token_id, int)
        self.assertIsInstance(tokenizer.unk_token_id, int)
        
        # Check that all special token IDs are different
        special_ids = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id
        ]
        self.assertEqual(len(special_ids), len(set(special_ids)))
    
    def test_tokenizer_training(self):
        """Test tokenizer training on sample texts."""
        tokenizer = BPETokenizer(
            vocab_size=300,  # Small vocab for testing
            special_tokens=self.special_tokens
        )
        
        initial_vocab_size = len(tokenizer.vocab)
        tokenizer.train(self.sample_texts)
        final_vocab_size = len(tokenizer.vocab)
        
        # Should have learned some merges
        self.assertGreater(final_vocab_size, initial_vocab_size)
        self.assertGreater(len(tokenizer.merges), 0)
    
    def test_encode_decode_cycle(self):
        """Test that encode/decode cycle preserves text."""
        tokenizer = BPETokenizer(
            vocab_size=500,
            special_tokens=self.special_tokens
        )
        tokenizer.train(self.sample_texts)
        
        for text in self.sample_texts:
            # Test basic encode/decode
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            self.assertEqual(decoded, text)
    
    def test_encode_with_special_tokens(self):
        """Test encoding with special tokens."""
        tokenizer = BPETokenizer(
            vocab_size=500,
            special_tokens=self.special_tokens
        )
        tokenizer.train(self.sample_texts)
        
        text = "Hello world"
        
        # Without special tokens
        tokens_no_special = tokenizer.encode(text, add_special_tokens=False)
        
        # With special tokens
        tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
        
        # Should have BOS and EOS added
        self.assertEqual(len(tokens_with_special), len(tokens_no_special) + 2)
        self.assertEqual(tokens_with_special[0], tokenizer.bos_token_id)
        self.assertEqual(tokens_with_special[-1], tokenizer.eos_token_id)
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special token handling."""
        tokenizer = BPETokenizer(
            vocab_size=500,
            special_tokens=self.special_tokens
        )
        tokenizer.train(self.sample_texts)
        
        text = "Hello world"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Decode with special tokens skipped
        decoded_skip = tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(decoded_skip, text)
        
        # Decode with special tokens included
        decoded_include = tokenizer.decode(tokens, skip_special_tokens=False)
        self.assertIn("<BOS>", decoded_include)
        self.assertIn("<EOS>", decoded_include)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        tokenizer = BPETokenizer(
            vocab_size=500,
            special_tokens=self.special_tokens
        )
        
        # Empty text without special tokens
        tokens = tokenizer.encode("", add_special_tokens=False)
        self.assertEqual(tokens, [])
        
        # Empty text with special tokens
        tokens = tokenizer.encode("", add_special_tokens=True)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0], tokenizer.bos_token_id)
        self.assertEqual(tokens[1], tokenizer.eos_token_id)
        
        # Decode empty
        decoded = tokenizer.decode([])
        self.assertEqual(decoded, "")
    
    def test_save_load_cycle(self):
        """Test saving and loading tokenizer."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        tokenizer.train(self.sample_texts)
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = Path(temp_dir) / "test_tokenizer.json"
        
        try:
            # Save tokenizer
            tokenizer.save(temp_file)
            self.assertTrue(temp_file.exists())
            
            # Create new tokenizer and load
            new_tokenizer = BPETokenizer(
                vocab_size=300,
                special_tokens=self.special_tokens
            )
            new_tokenizer.load(temp_file)
            
            # Test that both tokenizers produce same results
            test_text = "This is a test sentence."
            
            original_tokens = tokenizer.encode(test_text)
            loaded_tokens = new_tokenizer.encode(test_text)
            
            self.assertEqual(original_tokens, loaded_tokens)
            
            original_decoded = tokenizer.decode(original_tokens)
            loaded_decoded = new_tokenizer.decode(loaded_tokens)
            
            self.assertEqual(original_decoded, loaded_decoded)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_tokenize_method(self):
        """Test tokenize method that returns human-readable tokens."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        tokenizer.train(self.sample_texts)
        
        text = "Hello world"
        tokens = tokenizer.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # All tokens should be strings
        for token in tokens:
            self.assertIsInstance(token, str)
    
    def test_get_vocab_size(self):
        """Test vocabulary size reporting."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        
        initial_size = tokenizer.get_vocab_size()
        self.assertEqual(initial_size, 256 + len(self.special_tokens))
        
        tokenizer.train(self.sample_texts)
        final_size = tokenizer.get_vocab_size()
        
        self.assertGreaterEqual(final_size, initial_size)
    
    def test_invalid_vocab_size(self):
        """Test invalid vocabulary size handling."""
        with self.assertRaises(ValueError):
            BPETokenizer(vocab_size=0)
        
        with self.assertRaises(ValueError):
            BPETokenizer(vocab_size=-100)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        
        with self.assertRaises(FileNotFoundError):
            tokenizer.load("nonexistent_file.json")
    
    def test_training_with_empty_texts(self):
        """Test training with empty text list."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        
        with self.assertRaises(ValueError):
            tokenizer.train([])
    
    def test_training_with_min_frequency(self):
        """Test training with minimum frequency requirement."""
        tokenizer = BPETokenizer(
            vocab_size=300,
            special_tokens=self.special_tokens
        )
        
        # Train with high min_frequency - should stop early
        tokenizer.train(self.sample_texts, min_frequency=100)
        
        # Should have fewer merges than with default min_frequency
        self.assertLessEqual(len(tokenizer.merges), 50)


if __name__ == "__main__":
    unittest.main()