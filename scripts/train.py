#!/usr/bin/env python3
"""Simple training script that wraps the CLI interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


if __name__ == "__main__":
    from cli.train import main
    sys.exit(main())
