import os
from typing import List

import pandas as pd

from .config import CONFIG
from .utils import DatasetSpec, has_image_files


def _find_csv_by_keyword(files: List[str], keyword: str) -> str:
    for f in files:
        lf = f.lower()
        if lf.endswith(".csv") and keyword in lf:
            return f
    return ""


def inspect_dataset(data_dir: str) -> DatasetSpec:
    files = os.listdir(data_dir)
    train_file = _find_csv_by_keyword(files, "train") or _find_csv_by_keyword(files, "Train")
    test_file = _find_csv_by_keyword(files, "test") or _find_csv_by_keyword(files, "Test")
    sample_file = _find_csv_by_keyword(files, "sample") or _find_csv_by_keyword(files, "submission") or ""

    if not train_file:
        # fallback: first csv as train
        train_file = [f for f in files if f.endswith(".csv")][0]
    if not test_file:
        # fallback: second csv as test
        csvs = [f for f in files if f.endswith(".csv")]
        test_file = csvs[1] if len(csvs) > 1 else csvs[0]
    if not sample_file:
        # fallback: last csv as sample
        csvs = [f for f in files if f.endswith(".csv")]
        sample_file = csvs[-1]

    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)
    sample_path = os.path.join(data_dir, sample_file)

    df_train = pd.read_csv(train_path, nrows=CONFIG.max_rows_inspect)
    df_test = pd.read_csv(test_path, nrows=CONFIG.max_rows_inspect)

    # infer target column: columns in train not in test
    target_candidates = [c for c in df_train.columns if c not in df_test.columns]
    if not target_candidates:
        # fallback: last column
        target_column = df_train.columns[-1]
    else:
        target_column = target_candidates[-1]

    y = df_train[target_column]
    n_unique = y.nunique(dropna=True)
    n = len(y)

    # infer task type
    if y.dtype.kind in ("f", "c"):  # float or complex
        task_type = "regression"
    else:
        # heuristic: few unique  classification
        task_type = "classification" if n_unique <= max(20, 0.2 * n) else "regression"

    # infer modality
    if has_image_files(data_dir):
        modality = "image"
        long_text_cols = []
    else:
        obj_cols = df_train.select_dtypes(include=["object"]).columns.tolist()
        long_text_cols = []
        for col in obj_cols:
            sample_vals = df_train[col].dropna().astype(str).head(50)
            avg_len = sample_vals.map(len).mean() if len(sample_vals) > 0 else 0
            if avg_len > 20:  # heuristic
                long_text_cols.append(col)

        if long_text_cols:
            modality = "text"
        else:
            modality = "tabular"

    return DatasetSpec(
        data_dir=data_dir,
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        train_filename=train_file,
        test_filename=test_file,
        sample_submission_filename=sample_file,
        target_column=target_column,
        task_type=task_type,
        modality=modality,
        text_columns=long_text_cols if modality == "text" else [],
        image_column=None,
    )
