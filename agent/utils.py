import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DatasetSpec:
    data_dir: str
    train_path: str
    test_path: str
    sample_submission_path: str
    train_filename: str
    test_filename: str
    sample_submission_filename: str

    target_column: str
    task_type: str          # "classification" or "regression"
    modality: str           # "tabular", "text", "image", or "unknown"

    text_columns: List[str]
    image_column: Optional[str] = None


@dataclass
class Plan:
    script_type: str        # "tabular", "text", "image"
    model_name: str
    seed: int

    # data info (filenames kept for logging/debug)
    target_column: str
    task_type: str
    train_filename: str
    test_filename: str
    sample_submission_filename: str

    # NEW: absolute paths (what codegen should actually use)
    train_path: str = ""
    test_path: str = ""
    sample_submission_path: str = ""

    text_column: Optional[str] = None
    image_column: Optional[str] = None


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def has_image_files(root: str) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                return True
    return False
