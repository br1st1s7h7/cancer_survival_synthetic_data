from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_DIR = Path('data')
OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
TARGET = 'Survival_Status'

GROUPS = {
    'patient': ['Age', 'BMI', 'Sex', 'Ethnicity', 'Family_History_Cancer'],
    'cancer': ['Stage_at_Diagnosis', 'Tumor_Size_mm', 'Metastasis_Status', 'Genetic_Marker_Status', 'Cancer_Type'],
    'lifestyle': ['Smoking_Status', 'Alcohol_Consumption', 'Diabetes_Status', 'Obesity_Rate_Percent', 'Smoking_Prevalence_Percent'],
    'environment': ['AQI', 'PM2_5_ug_m3', 'PM10_ug_m3', 'Ozone_O3_ppb', 'NO2_ppb', 'SO2_ppb', 'CO_ppm'],
    'treatment': ['Treatment_Type', 'Chemotherapy_Cycles'],
    'healthcare': ['Insurance_Type', 'Hospital_Type', 'Distance_to_Treatment_Center_km'],
}
FEATURES = [c for cols in GROUPS.values() for c in cols]

EXTERNAL = [
    ('state_air_quality_index.csv', ['County_Year_Key'],
     ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ppb', 'Ozone_O3_ppb', 'SO2_ppb', 'CO_ppm', 'AQI']),
    ('state_obesity_rate.csv', ['County_Year_Key', 'Age_Group'], ['Obesity_Rate_Percent']),
    ('state_smoking_prevalence.csv', ['County_Year_Key', 'Age_Group', 'Sex'], ['Smoking_Prevalence_Percent']),
]


def read_csv(name):
    return pd.read_csv(DATA_DIR / name, keep_default_na=False, na_values=[''])


def load_data(dedupe=True):
    df = read_csv('cancer_patient_records_main.csv')
    if dedupe:
        df = df.drop_duplicates(subset='PatientID')
    df['Sex'] = df['Sex'].replace({'MALE': 'Male', 'FEMALE': 'Female'})
    df[TARGET] = df[TARGET].map({'Alive': 1, 'Dead': 0})
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    for name, keys, cols in EXTERNAL:
        table = read_csv(name)
        if 'Sex' in table.columns:
            table['Sex'] = table['Sex'].replace({'MALE': 'Male', 'FEMALE': 'Female'})
        table = table.dropna(subset=keys).drop_duplicates(subset=keys)[keys + cols]
        df = df.merge(table, on=keys, how='left')
    return df.reset_index(drop=True)


def get_data(dedupe=True, complete_case=False):
    df = load_data(dedupe)
    if complete_case:
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


def split(df, seed=SEED):
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=seed)


class Prep:
    def __init__(self, features=None, winsor=False, indicators=False):
        self.features = list(features or FEATURES)
        self.winsor = winsor
        self.indicators = indicators
        self.columns = None

    def fit(self, df):
        X = df[self.features]
        self.columns = None
        self.num = [c for c in self.features if pd.api.types.is_numeric_dtype(X[c])]
        self.cat = [c for c in self.features if c not in self.num]
        self.median = X[self.num].median()
        self.mode = {c: (X[c].mode().iloc[0] if X[c].notna().any() else 'Unknown') for c in self.cat}
        self.limits = {c: (X[c].quantile(0.01), X[c].quantile(0.99)) for c in self.num}
        self.flag = [c for c in self.features if X[c].isna().any()] if self.indicators else []
        self.columns = self.transform(df).columns.tolist()
        return self

    def transform(self, df):
        X = df[self.features]
        out = pd.DataFrame(index=X.index)
        for c in self.flag:
            out[c + '_missing'] = X[c].isna().astype(float)
        for c in self.num:
            v = X[c].fillna(self.median[c])
            if self.winsor:
                v = v.clip(*self.limits[c])
            out[c] = v.astype(float)
        for c in self.cat:
            out[c] = X[c].astype(object).where(X[c].notna(), self.mode[c]).astype(str)
        out = pd.get_dummies(out, columns=self.cat, dtype=float)
        if self.columns is not None:
            out = out.reindex(columns=self.columns, fill_value=0.0)
        return out

    def base_of(self, column):
        if column.endswith('_missing'):
            return column[:-8]
        for name in sorted(self.cat, key=len, reverse=True):
            if column.startswith(name + '_'):
                return name
        return column

    def columns_of(self, features):
        return [c for c in self.columns if self.base_of(c) in features]


def get_models(names=None, seed=SEED):
    models = {
        'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'DecisionTree': DecisionTreeClassifier(max_depth=8, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=seed),
        'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=seed),
        'CatBoost': CatBoostClassifier(iterations=300, learning_rate=0.05, verbose=0, random_seed=seed),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=seed),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(random_state=seed),
    }
    return {k: models[k] for k in names} if names else models


def evaluate(models, X_tr, y_tr, X_te, y_te):
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        rows.append({
            'model': name,
            'accuracy': accuracy_score(y_te, (p >= 0.5).astype(int)),
            'auc': roc_auc_score(y_te, p),
        })
    return pd.DataFrame(rows)


def rank_attributes(train, features=None):
    prep = Prep(features).fit(train)
    X = prep.transform(train)
    scaled = StandardScaler().fit_transform(X)
    weights = LogisticRegression(max_iter=2000).fit(scaled, train[TARGET]).coef_[0]
    series = pd.Series(np.abs(weights), index=X.columns)
    return series.groupby(prep.base_of).sum().sort_values(ascending=False)


def logreg_score(train, test, features, winsor=False, indicators=False):
    prep = Prep(features, winsor=winsor, indicators=indicators).fit(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(prep.transform(train), train[TARGET])
    p = model.predict_proba(prep.transform(test))[:, 1]
    return accuracy_score(test[TARGET], (p >= 0.5).astype(int)), roc_auc_score(test[TARGET], p)
