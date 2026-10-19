from nn_common import *

TOP = 10
SEEDS = [1, 2, 3, 4, 5]
NN_PAIR = ('Stage_at_Diagnosis', 'Treatment_Type')
PAIRS = [
    ('Stage_at_Diagnosis', 'Treatment_Type'),
    ('Stage_at_Diagnosis', 'Cancer_Type'),
    ('Age_Group', 'Stage_at_Diagnosis'),
    ('Age_Group', 'Cancer_Type'),
    ('Smoking_Status', 'Cancer_Type'),
    ('Smoking_Status', 'Age_Group'),
    ('Metastasis_Status', 'Cancer_Type'),
    ('Metastasis_Status', 'Stage_at_Diagnosis'),
]

df = get_data(complete_case=True)
df['BMI_Bin'] = pd.cut(df['BMI'], [0, 18.5, 25, 30, 100], labels=['under', 'normal', 'over', 'obese']).astype(str)
for a, b in PAIRS:
    df[f'{a}__x__{b}'] = df[a].astype(str) + '_' + df[b].astype(str)

extras = {f'{a} x {b}': [f'{a}__x__{b}'] for a, b in PAIRS}
extras['BMI bins'] = ['BMI_Bin']

rows = []
base_scores = {}
for seed in SEEDS:
    train, test = split(df, seed)
    base_features = rank_attributes(train, FEATURES).index[:TOP].tolist()
    base_scores[seed] = (base_features, logreg_score(train, test, base_features))

for label, columns in extras.items():
    deltas, accuracies = [], []
    for seed in SEEDS:
        train, test = split(df, seed)
        base_features, base = base_scores[seed]
        accuracy, auc = logreg_score(train, test, base_features + columns)
        deltas.append(auc - base[1])
        accuracies.append(accuracy - base[0])
    rows.append({'interaction': label, 'auc_change_mean': np.mean(deltas), 'auc_change_std': np.std(deltas),
                 'accuracy_change_mean': np.mean(accuracies), 'seeds_improved': int(np.sum(np.array(deltas) > 0))})

logistic = pd.DataFrame(rows).sort_values('auc_change_mean', ascending=False)
logistic.to_csv(OUT_DIR / 'e9_interactions_logistic.csv', index=False)
print(logistic.round(4).to_string(index=False))

pair_column = f'{NN_PAIR[0]}__x__{NN_PAIR[1]}'
nn_deltas = []
for seed in SEEDS:
    train, test = split(df, seed)
    base_features, _ = base_scores[seed]
    prep = Prep(base_features).fit(train)
    blocks = make_blocks(prep, base_features)
    base_auc, _ = train_and_score(blocks, prep.transform(train), train[TARGET], prep.transform(test), test[TARGET], seed)
    prep_x = Prep(base_features + [pair_column]).fit(train)
    blocks_x = make_blocks(prep_x, base_features)
    blocks_x['interaction'] = prep_x.columns_of([pair_column])
    new_auc, _ = train_and_score(blocks_x, prep_x.transform(train), train[TARGET], prep_x.transform(test), test[TARGET], seed)
    nn_deltas.append(new_auc - base_auc)

neural = pd.DataFrame([{
    'interaction': f'{NN_PAIR[0]} x {NN_PAIR[1]}',
    'auc_change_mean': np.mean(nn_deltas),
    'auc_change_std': np.std(nn_deltas),
    'seeds_improved': int(np.sum(np.array(nn_deltas) > 0)),
}])
neural.to_csv(OUT_DIR / 'e9_interactions_neural.csv', index=False)
print(neural.round(4).to_string(index=False))
