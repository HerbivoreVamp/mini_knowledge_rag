# 这里放=乱七八糟的小功能

from pathlib import Path


def clear_checkpoints(checkpoint_path: Path):
    if checkpoint_path.exists():
        checkpoint_path.unlink()
