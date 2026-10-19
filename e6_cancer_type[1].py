from common import *

MIN_TRAIN = 300
TOP = 5

df = get_data(complete_case=True)
train, test = split(df)
features = [c for c in FEATURES if c != 'Cancer_Type']

rows = []
for cancer_type, group in train.groupby('Cancer_Type'):
    test_group = test[test['Cancer_Type'] == cancer_type]
    if len(group) < MIN_TRAIN or group[TARGET].nunique() < 2 or test_group[TARGET].nunique() < 2:
        continue
    ranking = rank_attributes(group, features)
    accuracy, auc = logreg_score(group, test_group, features)
    rows.append({
        'cancer_type': cancer_type,
        'train_rows': len(group),
        'test_rows': len(test_group),
        'accuracy': accuracy,
        'auc': auc,
        'top_attributes': ', '.join(ranking.index[:TOP]),
    })

result = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
result.to_csv(OUT_DIR / 'e6_cancer_type.csv', index=False)
print(result.round(4).to_string(index=False))
