"""Unit tests for configuration system."""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt.config import (
    GPTConfig, ModelConfig, DataConfig, TrainingConfig, 
    TokenizerConfig, SamplingConfig, FilesConfig,
    load_config, save_config
)


class TestConfigSystem(unittest.TestCase):
    """Test configuration system functionality."""
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid config
        config = ModelConfig(
            max_block_size=512,
            vocab_size=50000,
            embed_dim=768,
            ff_dim=3072,
            num_layers=12,
            heads=12,
            dropout=0.1
        )
        self.assertEqual(config.embed_dim, 768)
        
        # Invalid: embed_dim not divisible by heads
        with self.assertRaises(ValueError):
            ModelConfig(
                max_block_size=512,
                vocab_size=50000,
                embed_dim=700,  # Not divisible by 12
                ff_dim=3072,
                num_layers=12,
                heads=12,
                dropout=0.1
            )
        
        # Invalid: negative values
        with self.assertRaises(ValueError):
            ModelConfig(
                max_block_size=-512,
                vocab_size=50000,
                embed_dim=768,
                ff_dim=3072,
                num_layers=12,
                heads=12,
                dropout=0.1
            )
    
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        # Valid config
        config = TrainingConfig(
            batch_size=32,
            learning_rate=5e-4,
            weight_decay=0.01,
            epochs=50
        )
        self.assertEqual(config.batch_size, 32)
        
        # Invalid: negative batch size
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=-1)
        
        # Invalid: negative learning rate
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=-0.001)
    
    def test_data_config_validation(self):
        """Test DataConfig validation."""
        # Valid config
        config = DataConfig(
            num_training_samples=100000,
            validation_split=0.1
        )
        self.assertEqual(config.num_training_samples, 100000)
        
        # Invalid: validation_split out of range
        with self.assertRaises(ValueError):
            DataConfig(validation_split=1.5)
        
        # Invalid: negative samples
        with self.assertRaises(ValueError):
            DataConfig(num_training_samples=-1000)
    
    def test_gpt_config_post_init(self):
        """Test GPTConfig post-initialization processing."""
        config = GPTConfig()
        
        # Check that tokenizer vocab_size was set automatically
        expected_tokenizer_vocab_size = config.model.vocab_size - len(config.tokenizer.special_tokens)
        self.assertEqual(config.tokenizer.vocab_size, expected_tokenizer_vocab_size)
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML."""
        # Create temporary YAML config
        config_data = {
            'model': {
                'max_block_size': 256,
                'vocab_size': 8192,
                'embed_dim': 256,
                'ff_dim': 1024,
                'num_layers': 6,
                'heads': 8,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'epochs': 20
            },
            'data': {
                'num_training_samples': 10000,
                'validation_split': 0.1
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            yaml.dump(config_data, temp_file, default_flow_style=False)
            temp_file.close()
            
            # Load config
            config = load_config(temp_file.name)
            
            self.assertEqual(config.model.max_block_size, 256)
            self.assertEqual(config.model.embed_dim, 256)
            self.assertEqual(config.training.batch_size, 16)
            self.assertEqual(config.data.num_training_samples, 10000)
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_yaml_config_saving(self):
        """Test saving configuration to YAML."""
        config = GPTConfig()
        config.model.embed_dim = 768  # Use 768 which is divisible by 12
        config.training.batch_size = 64
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_file.close()
        
        try:
            # Save config
            save_config(config, temp_file.name)
            
            # Load it back
            loaded_config = load_config(temp_file.name)
            
            self.assertEqual(loaded_config.model.embed_dim, 768)
            self.assertEqual(loaded_config.training.batch_size, 64)
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_partial_yaml_config(self):
        """Test loading partial YAML config with defaults."""
        # Only specify model config, others should use defaults
        config_data = {
            'model': {
                'embed_dim': 768,  # Use 768 which is divisible by 12
                'num_layers': 8
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            yaml.dump(config_data, temp_file, default_flow_style=False)
            temp_file.close()
            
            config = load_config(temp_file.name)
            
            # Specified values
            self.assertEqual(config.model.embed_dim, 768)
            self.assertEqual(config.model.num_layers, 8)
            
            # Default values should be used for unspecified fields
            self.assertEqual(config.model.max_block_size, 512)  # Default
            self.assertEqual(config.training.batch_size, 32)  # Default
            
        finally:
            Path(temp_file.name).unlink()
    
    def test_nonexistent_config_file(self):
        """Test loading from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_invalid_yaml_format(self):
        """Test loading invalid YAML."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            # Write invalid YAML
            temp_file.write("invalid: yaml: content: [unclosed")
            temp_file.close()
            
            with self.assertRaises(yaml.YAMLError):
                load_config(temp_file.name)
                
        finally:
            Path(temp_file.name).unlink()
    
    def test_config_field_types(self):
        """Test that config fields have correct types."""
        config = GPTConfig()
        
        # Model config
        self.assertIsInstance(config.model.max_block_size, int)
        self.assertIsInstance(config.model.dropout, float)
        
        # Training config
        self.assertIsInstance(config.training.batch_size, int)
        self.assertIsInstance(config.training.learning_rate, float)
        self.assertIsInstance(config.training.mixed_precision, bool)
        
        # Data config
        self.assertIsInstance(config.data.num_training_samples, int)
        self.assertIsInstance(config.data.validation_split, float)
    
    def test_special_tokens_config(self):
        """Test special tokens configuration."""
        custom_tokens = {
            "bos": "[BOS]",
            "eos": "[EOS]",
            "pad": "[PAD]",
            "unk": "[UNK]"
        }
        
        config = TokenizerConfig(special_tokens=custom_tokens)
        
        self.assertEqual(config.special_tokens["bos"], "[BOS]")
        self.assertEqual(config.special_tokens["eos"], "[EOS]")
        self.assertEqual(config.special_tokens["pad"], "[PAD]")
        self.assertEqual(config.special_tokens["unk"], "[UNK]")
    
    def test_files_config_directory_creation(self):
        """Test that FilesConfig creates directories."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = FilesConfig(
                log_dir=str(Path(temp_dir) / "test_logs"),
                output_dir=str(Path(temp_dir) / "test_outputs")
            )
            
            # Directories should be created during post_init
            self.assertTrue(Path(config.log_dir).exists())
            self.assertTrue(Path(config.output_dir).exists())
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()