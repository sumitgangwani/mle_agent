
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

DATA_DIR = r"C:\Users\sumit\Desktop\temp\hexo\tabular-playground-series-may-2022"
OUTPUT_DIR = r"."
TARGET_COL = "target"
TASK_TYPE = "classification"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

if TASK_TYPE == "classification":
    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
else:
    model = HistGradientBoostingRegressor(random_state=RANDOM_SEED)

pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED,
    stratify=y if TASK_TYPE == "classification" else None
)

pipeline.fit(X_train, y_train)

y_pred_val = pipeline.predict(X_val)

if TASK_TYPE == "classification":
    n_classes = len(np.unique(y))
    if n_classes == 2:
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

X_test = test.copy()
test_preds = pipeline.predict(X_test)

id_cols = [c for c in sample_sub.columns if c in test.columns]
target_cols = [c for c in sample_sub.columns if c not in id_cols]

submission = sample_sub.copy()
if len(target_cols) == 1:
    submission[target_cols[0]] = test_preds
else:
    if TASK_TYPE == "classification":
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
