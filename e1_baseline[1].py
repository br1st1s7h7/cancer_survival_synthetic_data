from common import *

df = get_data(dedupe=False)
train, test = split(df)
prep = Prep().fit(train)
models = get_models(['LogReg', 'XGBoost', 'DecisionTree', 'RandomForest', 'KNN', 'NaiveBayes'])
result = evaluate(models, prep.transform(train), train[TARGET], prep.transform(test), test[TARGET])
result.insert(0, 'rows', len(df))
result.to_csv(OUT_DIR / 'e1_baseline.csv', index=False)
print(result.round(4).to_string(index=False))
