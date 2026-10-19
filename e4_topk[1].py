from common import *

df = get_data(complete_case=True)
train, test = split(df)
ranking = rank_attributes(train)

frames = []
for k in [len(ranking), 10, 7, 5, 4, 3, 2, 1]:
    features = ranking.index[:k].tolist()
    prep = Prep(features).fit(train)
    result = evaluate(get_models(), prep.transform(train), train[TARGET], prep.transform(test), test[TARGET])
    result.insert(0, 'top_k', k)
    frames.append(result)

result = pd.concat(frames, ignore_index=True)
result.to_csv(OUT_DIR / 'e4_topk.csv', index=False)
print(result.pivot(index='model', columns='top_k', values='accuracy')[[len(ranking), 10, 7, 5, 4, 3, 2, 1]].round(4))
print(ranking.index[:10].tolist())
