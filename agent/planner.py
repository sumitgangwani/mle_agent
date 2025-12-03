from typing import Optional

from .utils import DatasetSpec, Plan


def make_plan(spec: DatasetSpec, seed: int) -> Plan:
    if spec.modality == "tabular":
        script_type = "tabular"
        model_name = "sklearn_tabular_gbdt"
        text_col: Optional[str] = None
        image_col: Optional[str] = None
    elif spec.modality == "text":
        script_type = "text"
        model_name = "sklearn_tfidf_logreg"
        text_col = spec.text_columns[0] if spec.text_columns else None
        image_col = None
    elif spec.modality == "image":
        script_type = "image"
        model_name = "torch_resnet18"
        text_col = None
        image_col = None
    else:
        script_type = "tabular"
        model_name = "sklearn_tabular_gbdt"
        text_col = None
        image_col = None

    return Plan(
        script_type=script_type,
        model_name=model_name,
        seed=seed,
        target_column=spec.target_column,
        task_type=spec.task_type,
        train_filename=spec.train_filename,
        test_filename=spec.test_filename,
        sample_submission_filename=spec.sample_submission_filename,
        text_column=text_col,
        image_column=image_col,
    )
