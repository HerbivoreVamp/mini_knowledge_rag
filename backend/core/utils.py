# 这里放一些通用功能
from pathlib import Path


def clear_checkpoints(checkpoint_path: Path) -> bool:
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    else:
        return False
