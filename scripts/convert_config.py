#!/usr/bin/env python3
"""Convert legacy config.json to new YAML format."""

import json
import yaml
import argparse
from pathlib import Path


def convert_config(json_path: str, yaml_path: str) -> None:
    """Convert JSON config to YAML format."""
    with open(json_path, 'r') as f:
        old_config = json.load(f)
    
    # Map old config structure to new structure
    new_config = {
        'data': {
            'num_training_samples': old_config['data']['num_training_samples'],
            'dataset_name': old_config['data']['dataset_name'],
            'dataset_config': old_config['data']['dataset_config'],
            'dataset_split': old_config['data']['dataset_split'],
            'data_file': old_config['data']['data_file'],
            'tokenizer_training_samples': old_config['data']['tokenizer_training_samples'],
            'validation_split': 0.1  # New field, set default
        },
        'model': {
            'max_block_size': old_config['model']['max_block_size'],
            'vocab_size': old_config['model']['vocab_size'],
            'embed_dim': old_config['model']['embed_dim'],
            'ff_dim': old_config['model']['ff_dim'],
            'num_layers': old_config['model']['num_layers'],
            'heads': old_config['model']['heads'],
            'dropout': old_config['model']['dropout']
        },
        'tokenizer': {
            'path': old_config['tokenizer']['path'],
            'special_tokens': {
                'bos': '<BOS>',
                'eos': '<EOS>',
                'pad': '<PAD>',
                'unk': '<UNK>'
            }
        },
        'training': {
            'batch_size': old_config['training']['batch_size'],
            'learning_rate': old_config['training']['learning_rate'],
            'weight_decay': old_config['training']['weight_decay'],
            'epochs': old_config['training']['epochs'],
            'max_grad_norm': old_config['training']['max_grad_norm'],
            'scheduler_T_max': old_config['training']['scheduler_T_max'],
            'scheduler_eta_min': old_config['training']['scheduler_eta_min'],
            'warmup_steps': 1000,  # New field, set default
            'gradient_accumulation_steps': 1,  # New field, set default
            'eval_interval': 1000,  # New field, set default
            'save_interval': 5000,  # New field, set default
            'early_stopping_patience': 10,  # New field, set default
            'mixed_precision': False  # New field, set default
        },
        'sampling': {
            'max_new_tokens': old_config['sampling']['max_new_tokens'],
            'temperature_default': old_config['sampling']['temperature_default'],
            'temperature_alt': old_config['sampling']['temperature_alt'],
            'top_k_default': old_config['sampling']['top_k_default'],
            'top_p_default': 0.95  # New field, set default
        },
        'files': {
            'checkpoint_path': old_config['files']['checkpoint_path'],
            'best_model_path': old_config['files']['best_model_path'],
            'log_dir': 'logs',  # New field, set default
            'output_dir': 'outputs'  # New field, set default
        }
    }
    
    # Save new YAML config
    with open(yaml_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)
    
    print(f"Converted {json_path} to {yaml_path}")


def main():
    """Main conversion entry point."""
    parser = argparse.ArgumentParser(description="Convert legacy JSON config to YAML")
    parser.add_argument(
        "--input", 
        type=str, 
        default="config.json",
        help="Input JSON config file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="config.yaml",
        help="Output YAML config file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return 1
    
    try:
        convert_config(args.input, args.output)
        return 0
    except Exception as e:
        print(f"Error converting config: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())