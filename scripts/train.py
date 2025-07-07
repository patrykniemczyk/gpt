#!/usr/bin/env python3
"""Simple training script that wraps the CLI interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.train import main

if __name__ == "__main__":
    sys.exit(main())