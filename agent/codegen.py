import os
from string import Template

from .utils import Plan


def _tabular_template() -> Template:
    return Template(
        """
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

RANDOM_SEED = $seed
np.random.seed(RANDOM_SEED)

DATA_DIR = r"$data_dir"
OUTPUT_DIR = r"$output_dir"
TARGET_COL = "$target_column"
TASK_TYPE = "$task_type"

train_path = os.path.join(DATA_DIR, "$train_filename")
test_path = os.path.join(DATA_DIR, "$test_filename")
sample_sub_path = os.path.join(DATA_DIR, "$sample_submission")

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
"""
    )


def _text_template() -> Template:
    return Template(
        """
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

RANDOM_SEED = $seed
np.random.seed(RANDOM_SEED)

DATA_DIR = r"$data_dir"
OUTPUT_DIR = r"$output_dir"
TARGET_COL = "$target_column"
TEXT_COL = "$text_column"

train_path = os.path.join(DATA_DIR, "$train_filename")
test_path = os.path.join(DATA_DIR, "$test_filename")
sample_sub_path = os.path.join(DATA_DIR, "$sample_submission")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

X_text = train[TEXT_COL].astype(str)
y = train[TARGET_COL]

# --- Handle pathological case: only one class in the target ---
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    const_class = unique_classes[0]
    print("VAL_METRIC: constant_target=1.0")

    id_cols = [c for c in sample_sub.columns if c in test.columns]
    target_cols = [c for c in sample_sub.columns if c not in id_cols]

    submission = sample_sub.copy()

    if len(target_cols) == 1:
        submission[target_cols[0]] = const_class
    else:
        for col in target_cols:
            if str(col) == str(const_class):
                submission[col] = 1.0
            else:
                submission[col] = 0.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")
    import sys
    sys.exit(0)

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
"""
    )


def _image_template() -> Template:
    return Template(
        """
import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = $seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_DIR = r"$data_dir"
OUTPUT_DIR = r"$output_dir"
TARGET_COL = "$target_column"

train_path = os.path.join(DATA_DIR, "$train_filename")
test_path = os.path.join(DATA_DIR, "$test_filename")
sample_sub_path = os.path.join(DATA_DIR, "$sample_submission")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

image_cols = [c for c in train_df.columns if "image" in c.lower() or "file" in c.lower()]
if image_cols:
    IMG_COL = image_cols[0]
else:
    shared = [c for c in train_df.columns if c in test_df.columns]
    IMG_COL = shared[0]

IMAGE_ROOT = DATA_DIR

class ImageDataset(Dataset):
    def __init__(self, df, img_col, targets=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col = img_col
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        possible_paths = [
            os.path.join(IMAGE_ROOT, img_name),
            os.path.join(IMAGE_ROOT, "train", img_name),
            os.path.join(IMAGE_ROOT, "images", img_name),
        ]
        img_path = None
        for p in possible_paths:
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Could not find image for {img_name}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.targets is None:
            return image, -1
        else:
            return image, self.targets[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y = train_df[TARGET_COL]
is_numeric = y.dtype.kind in ("i", "u", "f")
if is_numeric and y.nunique() > 20:
    task_type = "regression"
else:
    task_type = "classification"

if task_type == "classification":
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
else:
    le = None
    y_enc = y.values.astype("float32")
    num_classes = 1

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y if task_type == "classification" else None,
)
train_targets = y_enc[train_idx]
val_targets = y_enc[val_idx]

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

transform_train = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transform_val = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_ds = ImageDataset(train_split, IMG_COL, targets=train_targets, transform=transform_train)
val_ds = ImageDataset(val_split, IMG_COL, targets=val_targets, transform=transform_val)
test_ds = ImageDataset(test_df, IMG_COL, targets=None, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features
if task_type == "classification":
    model.fc = nn.Linear(in_features, num_classes)
else:
    model.fc = nn.Linear(in_features, 1)

model = model.to(device)

if task_type == "classification":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def run_epoch(loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    losses = []
    with torch.set_grad_enabled(train_mode):
        for x, yb in loader:
            x = x.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if task_type == "classification":
                loss = criterion(logits, yb.long())
            else:
                loss = criterion(logits.squeeze(), yb.float())
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)

num_epochs = 3
for epoch in range(num_epochs):
    train_loss = run_epoch(train_loader, train_mode=True)
    val_loss = run_epoch(val_loader, train_mode=False)
    print(f"Epoch {epoch+1}/{num_epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

print(f"VAL_METRIC: loss={val_loss}")

model.eval()
all_test_preds = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        logits = model(x)
        if task_type == "classification":
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_test_preds.append(probs)
        else:
            preds = logits.cpu().numpy()
            all_test_preds.append(preds)

if task_type == "classification":
    test_proba = np.concatenate(all_test_preds, axis=0)
else:
    test_proba = np.concatenate(all_test_preds, axis=0).squeeze()

id_cols = [c for c in sample_sub.columns if c in test_df.columns]
target_cols = [c for c in sample_sub.columns if c not in id_cols]
submission = sample_sub.copy()

if task_type == "classification":
    if len(target_cols) == test_proba.shape[1]:
        for i, col in enumerate(target_cols):
            submission[col] = test_proba[:, i]
    elif len(target_cols) == 1:
        if test_proba.ndim == 2 and test_proba.shape[1] >= 2:
            submission[target_cols[0]] = test_proba[:, 1]
        else:
            submission[target_cols[0]] = test_proba
    else:
        for col in target_cols:
            submission[col] = 0.0
else:
    if len(target_cols) == 1:
        submission[target_cols[0]] = test_proba
    else:
        for col in target_cols:
            submission[col] = test_proba

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}")
"""
    )


def generate_training_script(plan: Plan, data_dir: str, run_dir: str) -> str:
    if plan.script_type == "tabular":
        tmpl = _tabular_template()
    elif plan.script_type == "text":
        tmpl = _text_template()
    elif plan.script_type == "image":
        tmpl = _image_template()
    else:
        tmpl = _tabular_template()

    content = tmpl.substitute(
        seed=plan.seed,
        data_dir=data_dir,
        output_dir=".",
        target_column=plan.target_column,
        task_type=plan.task_type,
        train_filename=plan.train_filename,
        test_filename=plan.test_filename,
        sample_submission=plan.sample_submission_filename,
        text_column=plan.text_column or "",
    )

    script_path = os.path.join(run_dir, "generated_train.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    return script_path
