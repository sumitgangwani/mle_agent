
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

DATA_DIR = r"E:\projects\mle_agent\inputs\spooky_author"
OUTPUT_DIR = r"outputs\run_20251202_055506_seed0"
TARGET_COL = "author"
TEXT_COL = "text"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

X_text = train[TEXT_COL].astype(str)
y = train[TARGET_COL]

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

clf = LogisticRegression(
    max_iter=1000, multi_class="multinomial", solver="lbfgs", n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)

acc = accuracy_score(y_val, y_pred)
try:
    ll = log_loss(y_val, y_proba)
except Exception:
    ll = None

print(f"VAL_METRIC: accuracy={acc}, logloss={ll}")

X_test_text = test[TEXT_COL].astype(str)
X_test = vectorizer.transform(X_test_text)

id_cols = [c for c in sample_sub.columns if c in test.columns]
target_cols = [c for c in sample_sub.columns if c not in id_cols]

submission = sample_sub.copy()
proba_test = clf.predict_proba(X_test)

if len(target_cols) == proba_test.shape[1]:
    class_order = list(clf.classes_)
    for col in target_cols:
        if col in class_order:
            idx = class_order.index(col)
            submission[col] = proba_test[:, idx]
        else:
            submission[col] = 0.0
else:
    preds = clf.predict(X_test)
    if len(target_cols) == 1:
        submission[target_cols[0]] = preds
    else:
        for col in target_cols:
            submission[col] = (preds == col).astype(float)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")
