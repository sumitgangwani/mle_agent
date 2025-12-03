from dataclasses import dataclass


@dataclass
class GlobalConfig:
    val_size: float = 0.2
    random_seed: int = 42
    max_rows_inspect: int = 500  # for quick inspection, not full load


CONFIG = GlobalConfig()
