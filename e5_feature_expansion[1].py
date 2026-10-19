from common import *

TOP = 10
EXTRA = 8

df = get_data(complete_case=True)
train, test = split(df)
ranking = rank_attributes(train)

rows = []
for label, k in [('top_10', TOP), (f'top_10_plus_{EXTRA}', TOP + EXTRA), ('all_attributes', len(ranking))]:
    accuracy, auc = logreg_score(train, test, ranking.index[:k].tolist())
    rows.append({'feature_set': label, 'attributes': k, 'accuracy': accuracy, 'auc': auc})

result = pd.DataFrame(rows)
result.to_csv(OUT_DIR / 'e5_feature_expansion.csv', index=False)
print(result.round(4).to_string(index=False))
print(ranking.index[TOP:TOP + EXTRA].tolist())
