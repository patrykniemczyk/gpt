"""Unit tests for GPT model components."""

import unittest
import torch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt.model import GPT, SelfAttention, FeedForward, TransformerBlock
from gpt.config import ModelConfig


class TestModelComponents(unittest.TestCase):
    """Test individual model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 128
        self.heads = 8
        self.ff_dim = 512
        self.max_seq_len = 256
        self.batch_size = 2
        self.seq_len = 32
        self.vocab_size = 1000
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.sample_input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    
    def test_self_attention_creation(self):
        """Test SelfAttention module creation."""
        attention = SelfAttention(
            embed_dim=self.embed_dim,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        self.assertEqual(attention.embed_dim, self.embed_dim)
        self.assertEqual(attention.heads, self.heads)
        self.assertEqual(attention.head_dim, self.embed_dim // self.heads)
    
    def test_self_attention_forward(self):
        """Test SelfAttention forward pass."""
        attention = SelfAttention(
            embed_dim=self.embed_dim,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        output = attention(self.sample_input)
        
        self.assertEqual(output.shape, self.sample_input.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_self_attention_with_padding_mask(self):
        """Test SelfAttention with padding mask."""
        attention = SelfAttention(
            embed_dim=self.embed_dim,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        # Create padding mask (True for padding tokens)
        padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        padding_mask[:, -5:] = True  # Last 5 tokens are padding
        
        output = attention(self.sample_input, padding_mask)
        
        self.assertEqual(output.shape, self.sample_input.shape)
        # Check that padding positions are zeroed out
        self.assertTrue(torch.allclose(output[:, -5:], torch.zeros_like(output[:, -5:])))
    
    def test_feed_forward_creation(self):
        """Test FeedForward module creation."""
        ff = FeedForward(
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim
        )
        
        # Check structure
        self.assertEqual(len(ff.net), 4)  # Linear -> GELU -> Dropout -> Linear
    
    def test_feed_forward_forward(self):
        """Test FeedForward forward pass."""
        ff = FeedForward(
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim
        )
        
        output = ff(self.sample_input)
        
        self.assertEqual(output.shape, self.sample_input.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_transformer_block_creation(self):
        """Test TransformerBlock module creation."""
        block = TransformerBlock(
            embed_dim=self.embed_dim,
            heads=self.heads,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len
        )
        
        self.assertIsInstance(block.attention, SelfAttention)
        self.assertIsInstance(block.feed_forward, FeedForward)
    
    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass."""
        block = TransformerBlock(
            embed_dim=self.embed_dim,
            heads=self.heads,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len
        )
        
        output = block(self.sample_input)
        
        self.assertEqual(output.shape, self.sample_input.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_gpt_model_creation(self):
        """Test GPT model creation."""
        model = GPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_layers=4,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        self.assertEqual(model.vocab_size, self.vocab_size)
        self.assertEqual(model.embed_dim, self.embed_dim)
        self.assertEqual(len(model.layers), 4)
    
    def test_gpt_model_forward(self):
        """Test GPT model forward pass."""
        model = GPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_layers=2,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        logits = model(self.sample_input_ids)
        
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        self.assertFalse(torch.isnan(logits).any())
    
    def test_gpt_model_generation(self):
        """Test GPT model text generation."""
        model = GPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_layers=2,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        model.eval()
        
        # Generate from a simple input
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=False  # Use greedy decoding for deterministic test
            )
        
        self.assertEqual(generated.shape[0], 1)  # Batch size
        self.assertGreaterEqual(generated.shape[1], input_ids.shape[1])  # At least input length
        self.assertLessEqual(generated.shape[1], input_ids.shape[1] + 10)  # At most input + max_new_tokens
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = GPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_layers=2,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        total_params = model.get_num_params(non_embedding=False)
        non_embedding_params = model.get_num_params(non_embedding=True)
        
        self.assertGreater(total_params, 0)
        self.assertGreater(non_embedding_params, 0)
        self.assertLess(non_embedding_params, total_params)
    
    def test_model_save_load(self):
        """Test model save and load functionality."""
        import tempfile
        import shutil
        
        model = GPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_layers=2,
            heads=self.heads,
            max_seq_len=self.max_seq_len
        )
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save model
            model.save_pretrained(temp_dir)
            
            # Load model
            loaded_model = GPT.from_pretrained(temp_dir)
            
            # Test that models have same state dict keys
            original_keys = set(model.state_dict().keys())
            loaded_keys = set(loaded_model.state_dict().keys())
            self.assertEqual(original_keys, loaded_keys)
            
            # Test that models have same configuration
            self.assertEqual(model.vocab_size, loaded_model.vocab_size)
            self.assertEqual(model.embed_dim, loaded_model.embed_dim)
            self.assertEqual(model.num_layers, loaded_model.num_layers)
            
            # Test that trainable weights are properly loaded
            original_state = model.state_dict()
            loaded_state = loaded_model.state_dict()
            
            # Compare key weights (excluding buffers like causal_mask)
            trainable_keys = [
                'token_embedding.weight', 
                'position_embedding.weight',
                'layers.0.attention.q_proj.weight',
                'lm_head.weight'
            ]
            
            for key in trainable_keys:
                if key in original_state and key in loaded_state:
                    self.assertTrue(
                        torch.allclose(original_state[key], loaded_state[key]),
                        f"Weights don't match for {key}"
                    )
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_invalid_embed_dim_heads(self):
        """Test that invalid embed_dim/heads combination raises error."""
        with self.assertRaises(AssertionError):
            SelfAttention(
                embed_dim=127,  # Not divisible by heads
                heads=8,
                max_seq_len=self.max_seq_len
            )


if __name__ == "__main__":
    unittest.main()