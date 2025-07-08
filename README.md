# GPT

A minimal from-scratch implementation of the GPT architecture with BPE tokenization, training pipeline, and inference capabilities.

## Quick Start

### 1. Train a Model

Train a small model for testing:

```bash
python scripts/train.py --config configs/small.yaml --use-streaming
```

### 2. Generate Text

Interactive mode:

```bash
python scripts/inference.py --model-path outputs/checkpoints/best_model.pth --config configs/small.yaml --interactive
```

Single generation:

```bash
python scripts/inference.py --model-path outputs/checkpoints/best_model.pth --config configs/small.yaml --prompt "The future of AI"
```
