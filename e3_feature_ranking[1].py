from common import *
from sklearn.linear_model import LinearRegression

df = get_data(complete_case=True)
train, test = split(df)

logistic = rank_attributes(train)

prep = Prep().fit(train)
X = StandardScaler().fit_transform(prep.transform(train))
linear = pd.Series(np.abs(LinearRegression().fit(X, train[TARGET]).coef_), index=prep.columns)
linear = linear.groupby(prep.base_of).sum()

ranking = pd.DataFrame({'logistic_weight': logistic, 'linear_weight': linear.reindex(logistic.index)})
ranking.index.name = 'attribute'
ranking.to_csv(OUT_DIR / 'e3_feature_ranking.csv')
print(ranking.round(4).to_string())
