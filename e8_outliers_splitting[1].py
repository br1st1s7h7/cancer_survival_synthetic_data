from common import *
from sklearn.model_selection import StratifiedKFold

TOP = 18
SEEDS = [1, 2, 3, 4, 5]

df = get_data(complete_case=True)
ranking = rank_attributes(split(df)[0])
features = ranking.index[:TOP].tolist()
numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]


def iqr_keep(train):
    q1, q3 = train[numeric].quantile(0.25), train[numeric].quantile(0.75)
    spread = q3 - q1
    outside = (train[numeric] < q1 - 1.5 * spread) | (train[numeric] > q3 + 1.5 * spread)
    return train[~outside.any(axis=1)]


def z_keep(train):
    z = (train[numeric] - train[numeric].mean()) / train[numeric].std()
    return train[(z.abs() <= 3).all(axis=1)]


strategies = {
    'keep_outliers': (lambda t: t, False),
    'iqr_removal': (iqr_keep, False),
    'zscore_removal': (z_keep, False),
    'winsorization': (lambda t: t, True),
}

rows = []
for label, (reduce, winsor) in strategies.items():
    scores = []
    for seed in SEEDS:
        train, test = split(df, seed)
        reduced = reduce(train)
        scores.append(logreg_score(reduced, test, features, winsor=winsor) + (len(reduced),))
    scores = np.array(scores)
    rows.append({'strategy': label, 'split': 'stratified_holdout', 'accuracy': scores[:, 0].mean(),
                 'accuracy_std': scores[:, 0].std(), 'auc': scores[:, 1].mean(), 'train_rows': scores[:, 2].mean()})

folds = []
for seed in SEEDS:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kfold.split(df, df[TARGET]):
        folds.append(logreg_score(df.iloc[tr_idx], df.iloc[te_idx], features, winsor=True))
folds = np.array(folds)
rows.append({'strategy': 'winsorization', 'split': 'stratified_5fold', 'accuracy': folds[:, 0].mean(),
             'accuracy_std': folds[:, 0].std(), 'auc': folds[:, 1].mean(), 'train_rows': len(df) * 0.8})

result = pd.DataFrame(rows)
result.to_csv(OUT_DIR / 'e8_outliers_splitting.csv', index=False)
print(result.round(4).to_string(index=False))
