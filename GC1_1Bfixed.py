import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix

REAL_FILE = "cancer_patient_records_main.csv"
SYNTHETIC_FILE = "synthetic_1b_gaussian_copula_1M_fixed.csv"
TARGET = "Survival_Status"

real = pd.read_csv(REAL_FILE)
synth = pd.read_csv(SYNTHETIC_FILE)

def encode_target(s):
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().map({"alive": 1, "dead": 0})
    return s

real[TARGET] = encode_target(real[TARGET])
synth[TARGET] = encode_target(synth[TARGET])
real = real.dropna(subset=[TARGET])
synth = synth.dropna(subset=[TARGET])

_, real_test = train_test_split(
    real, test_size=0.2, stratify=real[TARGET], random_state=42
)

features = [c for c in synth.columns if c != TARGET and c in real_test.columns]

X_train, y_train = synth[features], synth[TARGET].astype(int)
X_test, y_test = real_test[features], real_test[TARGET].astype(int)

num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegressionCV(
        Cs=5, cv=3, solver="lbfgs", max_iter=1000, n_jobs=-1, scoring="roc_auc"
    )),
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))