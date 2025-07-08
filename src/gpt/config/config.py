"""Configuration classes with type validation and YAML support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    num_training_samples: int = 100000
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "CC-MAIN-2014-10"
    dataset_split: str = "train"
    data_file: str = "data.json"
    tokenizer_training_samples: int = 10000
    validation_split: float = 0.1

    def __post_init__(self):
        """Validate configuration values."""
        if self.num_training_samples <= 0:
            raise ValueError("num_training_samples must be positive")
        if self.tokenizer_training_samples <= 0:
            raise ValueError("tokenizer_training_samples must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    max_block_size: int = 512
    vocab_size: int = 50000
    embed_dim: int = 768
    ff_dim: int = 3072
    num_layers: int = 12
    heads: int = 12
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_block_size <= 0:
            raise ValueError("max_block_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.embed_dim % self.heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer."""

    path: str = "tokenizer.txt"
    vocab_size: Optional[int] = None  # Will use model.vocab_size - 3 if None
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "bos": "<BOS>",
            "eos": "<EOS>",
            "pad": "<PAD>",
            "unk": "<UNK>",
        }
    )

    def __post_init__(self):
        """Validate configuration values."""
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 50
    max_grad_norm: float = 1.0
    scheduler_T_max: int = 1000
    scheduler_eta_min: float = 1e-6
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    eval_interval: int = 1000
    save_interval: int = 5000
    early_stopping_patience: int = 10
    mixed_precision: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass
class SamplingConfig:
    """Configuration for text sampling."""

    max_new_tokens: int = 512
    temperature_default: float = 1.0
    temperature_alt: float = 0.8
    top_k_default: int = 50
    top_p_default: float = 0.95

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature_default <= 0:
            raise ValueError("temperature_default must be positive")
        if self.temperature_alt <= 0:
            raise ValueError("temperature_alt must be positive")
        if self.top_k_default <= 0:
            raise ValueError("top_k_default must be positive")
        if not 0 <= self.top_p_default <= 1:
            raise ValueError("top_p_default must be between 0 and 1")


@dataclass
class FilesConfig:
    """Configuration for file paths."""

    checkpoint_path: str = "checkpoint.pth"
    best_model_path: str = "best_model.pth"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    def __post_init__(self):
        """Validate and create directories if needed."""
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.output_dir).mkdir(exist_ok=True)


@dataclass
class GPTConfig:
    """Main configuration class combining all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    files: FilesConfig = field(default_factory=FilesConfig)

    def __post_init__(self):
        """Post-process configuration after initialization."""
        # Set tokenizer vocab_size if not specified
        if self.tokenizer.vocab_size is None:
            self.tokenizer.vocab_size = self.model.vocab_size - len(
                self.tokenizer.special_tokens
            )


def load_config(config_path: Union[str, Path]) -> GPTConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        GPTConfig: Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")

    if config_dict is None:
        config_dict = {}

    # Convert nested dicts to dataclass instances
    config_kwargs = {}
    for field_name, field_type in [
        ("data", DataConfig),
        ("model", ModelConfig),
        ("tokenizer", TokenizerConfig),
        ("training", TrainingConfig),
        ("sampling", SamplingConfig),
        ("files", FilesConfig),
    ]:
        if field_name in config_dict:
            config_kwargs[field_name] = field_type(**config_dict[field_name])

    return GPTConfig(**config_kwargs)


def save_config(config: GPTConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    config_dict = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "tokenizer": config.tokenizer.__dict__,
        "training": config.training.__dict__,
        "sampling": config.sampling.__dict__,
        "files": config.files.__dict__,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
