"""Entrypoint for synchronous distributed language model training and aggregation."""

import os
import sys

# Ensure src/ is on sys.path if package is run directly from checkout without installation
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import tplr
from training import (
    DistributedModelTrainer,
    Timer,
    dict_parser_type,
    load_experiment_config as get_config,
    str_to_bool,
)

__all__ = [
    "DistributedModelTrainer",
    "Timer",
    "dict_parser_type",
    "get_config",
    "main",
    "str_to_bool",
]


def main():
    trainer = DistributedModelTrainer()
    if getattr(trainer.config, "debug", False):
        tplr.debug()
    if getattr(trainer.config, "trace", False):
        tplr.trace()

    try:
        trainer.run()
    except KeyboardInterrupt:
        tplr.logger.info("Training interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
