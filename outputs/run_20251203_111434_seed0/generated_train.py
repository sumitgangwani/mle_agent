
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

DATA_DIR = r"E:\projects\mle_agent\inputs\text_normalization"
OUTPUT_DIR = r"."
TARGET_COL = "after"
TASK_TYPE = "regression"

train_path = os.path.join(DATA_DIR, "en_train.csv")
test_path = os.path.join(DATA_DIR, "en_test_2.csv")
sample_sub_path = os.path.join(DATA_DIR, "en_sample_submission_2.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

# --- Feature selection: only columns present in BOTH train and test ---
X_full = train.drop(columns=[TARGET_COL])
shared_feature_cols = [c for c in X_full.columns if c in test.columns]

if not shared_feature_cols:
    raise RuntimeError("No shared feature columns between train and test after dropping target.")

X = train[shared_feature_cols]
y = train[TARGET_COL]

# --- Drop rows with missing target values (generic NaN-safe handling) ---
mask = y.notna()
if mask.sum() < len(y):
    X = X[mask]
    y = y[mask]

# --- If target is non-numeric, force classification at runtime ---
if y.dtype.kind not in ("i", "u", "f"):
    runtime_task_type = "classification"
else:
    runtime_task_type = TASK_TYPE

X_test = test[shared_feature_cols]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ),
        ("num", "passthrough", num_cols),
    ]
)

if runtime_task_type == "classification":
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
else:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

# --- Safe stratification: only if all classes have >= 2 samples ---
if runtime_task_type == "classification":
    vc = pd.Series(y).value_counts()
    if (vc < 2).any():
        stratify_param = None
    else:
        stratify_param = y
else:
    stratify_param = None

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify_param
)

pipeline.fit(X_train, y_train)

y_pred_val = pipeline.predict(X_val)

if runtime_task_type == "classification":
    n_classes = len(np.unique(y))
    if n_classes == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
        proba = pipeline.predict_proba(X_val)[:, 1]
        metric = roc_auc_score(y_val, proba)
        metric_name = "roc_auc"
    else:
        metric = accuracy_score(y_val, y_pred_val)
        metric_name = "accuracy"
else:
    metric = mean_squared_error(y_val, y_pred_val, squared=False)
    metric_name = "rmse"

print(f"VAL_METRIC: {metric_name}={metric}")

# --- Predict on test using the SAME feature columns ---
test_preds = pipeline.predict(X_test)

id_cols = [c for c in sample_sub.columns if c in test.columns]
target_cols = [c for c in sample_sub.columns if c not in id_cols]

submission = sample_sub.copy()
if len(target_cols) == 1:
    submission[target_cols[0]] = test_preds
else:
    if runtime_task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] == len(target_cols):
            for i, col in enumerate(target_cols):
                submission[col] = proba[:, i]
        else:
            for col in target_cols:
                submission[col] = (test_preds == col).astype(float)
    else:
        for col in target_cols:
            submission[col] = test_preds

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")
