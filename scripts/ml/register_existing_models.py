#!/usr/bin/env python3
"""
Register Existing Models in the Model Registry.

Scans data/models/ for trained models and registers them in the model registry
so they can be used by the trading system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def scan_models_directory(models_dir: Path) -> List[Dict]:
    """Scan the models directory for existing trained models."""
    found_models = []

    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return found_models

    # Look for model files with pattern: {symbol}_{model_type}_model.pkl
    for model_file in models_dir.glob("*_model.pkl"):
        name = model_file.stem.replace("_model", "")
        parts = name.split("_")

        if len(parts) >= 3:
            # e.g., BTC_USDT_random_forest
            symbol = f"{parts[0]}/{parts[1]}"
            model_type = "_".join(parts[2:])

            # Look for metadata
            meta_file = model_file.parent / f"{name}_meta.json"
            accuracy = 0.55
            created_at = datetime.now()

            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        accuracy = meta.get("accuracy", meta.get("val_accuracy", 0.55))
                        if "trained_at" in meta:
                            created_at = datetime.fromisoformat(meta["trained_at"])
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")

            found_models.append({
                "name": name,
                "symbol": symbol,
                "model_type": model_type,
                "path": str(model_file.parent),
                "accuracy": accuracy,
                "created_at": created_at,
            })
            logger.info(f"Found model: {name} ({model_type}) for {symbol}")

    # Look for directory-based models (subdirectories)
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            # Check for model files inside
            for pattern in ["model.pkl", "model.pt", "*.pkl"]:
                model_files = list(subdir.glob(pattern))
                if model_files:
                    # Parse directory name
                    dir_name = subdir.name
                    parts = dir_name.split("_")

                    if len(parts) >= 4:
                        # e.g., BTC_USDT_1h_random_forest
                        symbol = f"{parts[0]}/{parts[1]}"
                        timeframe = parts[2] if parts[2] in ["1h", "4h", "1d"] else None
                        model_type = "_".join(parts[3:]) if timeframe else "_".join(parts[2:])

                        # Check for metadata
                        meta_file = subdir / "meta.json"
                        accuracy = 0.55
                        created_at = datetime.now()

                        if meta_file.exists():
                            try:
                                with open(meta_file) as f:
                                    meta = json.load(f)
                                    accuracy = meta.get("accuracy", 0.55)
                            except (json.JSONDecodeError, IOError, KeyError):
                                pass

                        found_models.append({
                            "name": dir_name,
                            "symbol": symbol,
                            "model_type": model_type,
                            "path": str(subdir),
                            "accuracy": accuracy,
                            "created_at": created_at,
                        })
                        logger.info(f"Found model: {dir_name} ({model_type}) for {symbol}")
                    break

    return found_models


def register_models_in_registry(models: List[Dict], registry_dir: Path):
    """Register found models in the model registry."""
    registry_file = registry_dir / "registry.json"

    # Load existing registry
    existing_registry = {"models": {}}
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                data = json.load(f)
                if "models" in data:
                    existing_registry = data
                else:
                    # Old format - convert
                    existing_registry = {"models": data}
        except Exception as e:
            logger.warning(f"Could not load existing registry: {e}")
            existing_registry = {"models": {}}

    registered_count = 0

    for model in models:
        key = f"{model['symbol'].replace('/', '_')}_{model['model_type']}"

        # Determine market type
        symbol = model["symbol"]
        if "/" in symbol:
            base = symbol.split("/")[0]
            if base in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE", "ADA"]:
                market_type = "crypto"
            elif base in ["XAU", "XAG", "USOIL", "NATGAS"]:
                market_type = "commodity"
            else:
                market_type = "stock"
        else:
            market_type = "stock"

        # Create registry entry
        entry = {
            "name": model["name"],
            "model_type": model["model_type"],
            "version": "1.0.0",
            "market_type": market_type,
            "symbol": model["symbol"],
            "created_at": model["created_at"].isoformat(),
            "accuracy": model["accuracy"],
            "val_accuracy": model["accuracy"],
            "path": model["path"],
            "is_active": True,
            "metadata": {},
        }

        existing_registry["models"][key] = entry
        registered_count += 1
        logger.info(f"Registered: {key} ({market_type})")

    # Save registry
    registry_dir.mkdir(parents=True, exist_ok=True)
    with open(registry_file, "w") as f:
        json.dump(existing_registry, f, indent=2)

    logger.info(f"Registered {registered_count} models in {registry_file}")
    return registered_count


def main():
    """Main entry point."""
    models_dir = Path("data/models")
    registry_dir = Path("data/model_registry")

    logger.info("Scanning for existing trained models...")
    models = scan_models_directory(models_dir)

    if not models:
        logger.warning("No models found in data/models/")
        logger.info("Run scripts/ml/quick_train.py to train some models first")
        return

    logger.info(f"\nFound {len(models)} models")

    logger.info("\nRegistering models in registry...")
    count = register_models_in_registry(models, registry_dir)

    logger.info(f"\nDone! {count} models registered.")
    logger.info("Models will now be available for the trading system.")


if __name__ == "__main__":
    main()
