from nn_common import *

TOP = 10
SEEDS = [1, 2, 3, 4, 5]
BASE_BLOCKS = ['cancer', 'patient']

df = get_data(complete_case=True)
ranking = rank_attributes(split(df)[0])
features = ranking.index[:TOP].tolist()


def run(block_names, seed, flat=False):
    train, test = split(df, seed)
    prep = Prep(features).fit(train)
    X_tr, X_te = prep.transform(train), prep.transform(test)
    blocks = make_blocks(prep, features, block_names)
    if flat:
        blocks = {'all': [c for cols in blocks.values() for c in cols]}
    return train_and_score(blocks, X_tr, train[TARGET], X_te, test[TARGET], seed)


present = [n for n in GROUPS if any(c in features for c in GROUPS[n])]
base = [b for b in BASE_BLOCKS if b in present]
setups = {'+'.join(base): base}
for extra in present:
    if extra not in base:
        setups['+'.join(base + [extra])] = base + [extra]
setups['all_blocks'] = present

rows = []
for label, names in setups.items():
    scores = np.array([run(names, s) for s in SEEDS])
    rows.append({'blocks': label, 'auc_mean': scores[:, 0].mean(), 'auc_std': scores[:, 0].std(),
                 'acc_mean': scores[:, 1].mean(), 'acc_std': scores[:, 1].std()})

scores = np.array([run(present, s, flat=True) for s in SEEDS])
rows.append({'blocks': 'flat_network', 'auc_mean': scores[:, 0].mean(), 'auc_std': scores[:, 0].std(),
             'acc_mean': scores[:, 1].mean(), 'acc_std': scores[:, 1].std()})

result = pd.DataFrame(rows)
result.to_csv(OUT_DIR / 'e7_branch_nn.csv', index=False)
print(result.round(4).to_string(index=False))
