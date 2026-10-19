import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

real_path = "cancer_patient_records_main.csv"
synthetic_paths = {
    "syn1": "synthetic_1a_gaussian_copula_1M.csv",
    "syn2": "synthetic_1b_gaussian_copula_1M.csv",
    "syn3": "synthetic_2a_gaussian_copula_1M.csv",
    "syn4": "synthetic_2b_gaussian_copula_1M.csv",
    "syn5": "synthetic_3a_gaussian_copula_1M.csv",
    "syn6": "synthetic_3b_gaussian_copula_1M.csv",
}

drop_cols = [
    "Record_ID",
    "PatientID",
    "Location_ID",
    "County_Year_Key",
    "State_Year_Key",
    "Survey_Batch_ID",
    "Data_Entry_Clerk_ID",
    "Notes",
    "Diagnosis_Date",
]

max_categories = 50
syn_train_rows = 200000
test_size = 0.3
n_seeds = 3
seed = 42


def clean(d):
    d = d.drop(columns=drop_cols, errors="ignore")
    if "Sex" in d.columns:
        d["Sex"] = d["Sex"].replace({"MALE": "Male", "FEMALE": "Female"})
    return d


real = clean(pd.read_csv(real_path))

target = None
for c in real.columns:
    if not pd.api.types.is_numeric_dtype(real[c]):
        vals = {str(v).lower() for v in real[c].dropna().unique()}
        if {"alive", "dead"} <= vals:
            target = c

num_f = [
    c for c in real.columns
    if c != target and pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
]
cat_f = [
    c for c in real.columns
    if c != target and c not in num_f and real[c].nunique() <= max_categories
]
feats = num_f + cat_f


def prep(d):
    d = d.dropna(subset=[target]).reset_index(drop=True)
    X = d.reindex(columns=feats).copy()
    for c in num_f:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_f:
        X[c] = X[c].astype(str)
    y = (d[target].astype(str).str.lower() == "dead").astype(int).to_numpy()
    return X, y


def build(rs):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), num_f),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_f),
    ])
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=rs,
    )
    return Pipeline([("pre", pre), ("mlp", mlp)])


def evaluate(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_te, proba),
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
    }


X_all, y_all = prep(real)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=test_size, stratify=y_all, random_state=seed
)

rows = []

for s in range(n_seeds):
    model = build(seed + s)
    model.fit(X_tr, y_tr)
    res = evaluate(model, X_te, y_te)
    res.update({"train_data": "real_baseline", "train_pos_rate": y_tr.mean(), "seed": s})
    rows.append(res)

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    syn = syn.dropna(subset=[target])
    if len(syn) > syn_train_rows:
        syn = syn.sample(n=syn_train_rows, random_state=seed)
    Xs, ys = prep(syn)
    for s in range(n_seeds):
        model = build(seed + s)
        model.fit(Xs, ys)
        res = evaluate(model, X_te, y_te)
        res.update({"train_data": name, "train_pos_rate": ys.mean(), "seed": s})
        rows.append(res)

results = pd.DataFrame(rows)
metric_cols = ["auc", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "train_pos_rate"]
summary = results.groupby("train_data")[metric_cols].mean()
auc_std = results.groupby("train_data")["auc"].std().rename("auc_std")
summary = summary.join(auc_std)

base = summary.loc["real_baseline"]
summary["auc_ratio"] = summary["auc"] / base["auc"]
summary["auc_gap"] = base["auc"] - summary["auc"]
summary["f1_ratio"] = summary["f1"] / base["f1"] if base["f1"] > 0 else np.nan

print("test_pos_rate", round(y_te.mean(), 4), "test_rows", len(y_te))
print(summary.sort_values("auc", ascending=False).round(4))