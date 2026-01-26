#!/usr/bin/env python3
"""Quick training for AVAX/USDT ensemble"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Run the improved_training script with AVAX symbol
if __name__ == "__main__":
    import subprocess
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab"
    result = subprocess.run(
        [sys.executable, "scripts/ml/improved_training.py", "--symbols", "AVAX/USDT", "--days", "730"],
        cwd="/Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab",
        env=env
    )
    sys.exit(result.returncode)
