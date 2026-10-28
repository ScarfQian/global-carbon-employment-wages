#!/usr/bin/env python3
"""Summarize numerical results without rendering figures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carbon_labour.summaries import main


if __name__ == "__main__":
    main()
