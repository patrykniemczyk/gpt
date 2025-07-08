"""Command line interface for inference with GPT models."""

import argparse
import sys
from pathlib import Path

import torch

from gpt.utils.logging import setup_logging, get_logger
from gpt.tokenizer import BPETokenizer
from gpt.model import GPT
from gpt.config import load_config

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint or directory",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        help="Path to tokenizer file (will try to infer from config if not provided)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (will try to infer from model directory if not provided)",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument(
        "--top-p", type=float, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, console_output=True)
    logger = get_logger(__name__)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Using device: %s", device)

    try:
        # Load model and tokenizer
        model, tokenizer, _ = load_model_and_tokenizer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            config_path=args.config,
            device=device,
        )

        logger.info("Model and tokenizer loaded successfully")
        logger.info("Model has %s parameters", f"{model.get_num_params():,}")

        if args.interactive:
            # Interactive mode
            run_interactive_inference(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        else:
            # Single generation
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )

            print("Generated text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)

        return 0

    except Exception as e:
        logger.error("Inference failed: %s", e)
        return 1


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str = None,
    config_path: str = None,
    device: torch.device = None,
):
    """Load model, tokenizer, and configuration."""
    model_path = Path(model_path)

    # Try to load as a saved model directory first
    if model_path.is_dir():
        # Load from directory
        model = GPT.from_pretrained(str(model_path))

        # Try to find tokenizer in the same directory
        if tokenizer_path is None:
            tokenizer_candidates = [
                model_path / "tokenizer.json",
                model_path / "tokenizer.txt",
                model_path.parent / "tokenizer.json",
                model_path.parent / "tokenizer.txt",
            ]
            for candidate in tokenizer_candidates:
                if candidate.exists():
                    tokenizer_path = str(candidate)
                    break

        # Try to find config in the same directory
        if config_path is None:
            config_candidates = [
                model_path / "config.yaml",
                model_path.parent / "config.yaml",
                model_path.parent / "configs" / "default.yaml",
            ]
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break

    else:
        # Load from checkpoint file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # We need config to create the model
        if config_path is None:
            # Try to find config in parent directories
            config_candidates = [
                model_path.parent / "config.yaml",
                model_path.parent.parent / "config.yaml",
                model_path.parent / "configs" / "default.yaml",
            ]
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = str(candidate)
                    break

        if config_path is None:
            raise ValueError("Config file is required when loading from checkpoint")

        # Load config and create model
        config = load_config(config_path)

        # We need tokenizer to determine vocab size
        if tokenizer_path is None:
            tokenizer_candidates = [
                model_path.parent / "tokenizer.json",
                model_path.parent / "tokenizer.txt",
                Path(config.tokenizer.path),
            ]
            for candidate in tokenizer_candidates:
                if candidate.exists():
                    tokenizer_path = str(candidate)
                    break

        if tokenizer_path is None:
            raise ValueError("Tokenizer file is required when loading from checkpoint")

        # Load tokenizer first to get vocab size
        tokenizer = BPETokenizer(special_tokens=config.tokenizer.special_tokens)
        tokenizer.load(tokenizer_path)

        # Create model
        model = GPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=config.model.embed_dim,
            ff_dim=config.model.ff_dim,
            num_layers=config.model.num_layers,
            heads=config.model.heads,
            dropout=config.model.dropout,
            max_seq_len=config.model.max_block_size,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Load tokenizer if not already loaded
    if "tokenizer" not in locals():
        if tokenizer_path is None:
            raise ValueError("Tokenizer path is required")

        # Load config if not already loaded to get special tokens
        if "config" not in locals():
            if config_path:
                config = load_config(config_path)
                special_tokens = config.tokenizer.special_tokens
            else:
                special_tokens = {
                    "bos": "<BOS>",
                    "eos": "<EOS>",
                    "pad": "<PAD>",
                    "unk": "<UNK>",
                }
        else:
            special_tokens = config.tokenizer.special_tokens

        tokenizer = BPETokenizer(special_tokens=special_tokens)
        tokenizer.load(tokenizer_path)

    # Load config if not already loaded
    if "config" not in locals():
        config = None

    # Move model to device
    if device:
        model = model.to(device)

    model.eval()

    return model, tokenizer, config


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int = None,
    top_p: float = None,
    device: torch.device = None,
) -> str:
    """Generate text from a prompt."""
    # Encode prompt
    if prompt:
        input_ids = tokenizer.encode(prompt)
    else:
        input_ids = [tokenizer.bos_token_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    if device:
        input_tensor = input_tensor.to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    return generated_text


def run_interactive_inference(
    model: GPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    max_tokens: int,
    temperature: float,
    top_k: int = None,
    top_p: float = None,
):
    """Run interactive inference loop."""
    print("Interactive GPT Inference")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ["quit", "exit"]:
                break
            if prompt.lower() == "help":
                print("\nCommands:")
                print("  help - Show this help message")
                print("  quit/exit - Exit the program")
                print("  Any other text will be used as a prompt for generation")
                continue
            if not prompt:
                continue

            print("\nGenerating...")
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

            print("\nGenerated text:")
            print("-" * 30)
            print(generated_text)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")


if __name__ == "__main__":
    sys.exit(main())
