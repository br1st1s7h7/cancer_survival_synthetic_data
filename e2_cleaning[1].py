from common import *

versions = {
    'raw': get_data(dedupe=False),
    'duplicates_removed': get_data(dedupe=True),
    'duplicates_and_missing_removed': get_data(dedupe=True, complete_case=True),
}
names = ['LogReg', 'XGBoost', 'DecisionTree', 'RandomForest', 'KNN', 'NaiveBayes']

frames = []
for label, df in versions.items():
    train, test = split(df)
    prep = Prep().fit(train)
    result = evaluate(get_models(names), prep.transform(train), train[TARGET], prep.transform(test), test[TARGET])
    result.insert(0, 'rows', len(df))
    result.insert(0, 'data_version', label)
    frames.append(result)

result = pd.concat(frames, ignore_index=True)
result.to_csv(OUT_DIR / 'e2_cleaning.csv', index=False)
print(result.pivot(index='model', columns='data_version', values='accuracy').round(4))
print(result.groupby('data_version')['rows'].first())
