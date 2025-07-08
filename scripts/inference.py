#!/usr/bin/env python3
"""Simple inference script that wraps the CLI interface."""

from cli.inference import main
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


if __name__ == "__main__":
    sys.exit(main())
