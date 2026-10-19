import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

real_path = "cancer_patient_records_main.csv"
synthetic_paths = {
    "syn1": "synthetic_1a_gaussian_copula_1M.csv",
    "syn2": "synthetic_1b_gaussian_copula_1M.csv",
    "syn3": "synthetic_2a_gaussian_copula_1M.csv",
    "syn4": "synthetic_2b_gaussian_copula_1M.csv",
    "syn5": "synthetic_3a_gaussian_copula_1M.csv",
    "syn6": "synthetic_3b_gaussian_copula_1M.csv",
}

drop_cols = [
    "Record_ID",
    "PatientID",
    "Location_ID",
    "County_Year_Key",
    "State_Year_Key",
    "Survey_Batch_ID",
    "Data_Entry_Clerk_ID",
    "Notes",
    "Diagnosis_Date",
]

max_categories = 50
triple_max_levels = 12
n_proj = 100
syn_sample = 200000
seed = 0


def clean(d):
    d = d.drop(columns=drop_cols, errors="ignore")
    if "Sex" in d.columns:
        d["Sex"] = d["Sex"].replace({"MALE": "Male", "FEMALE": "Female"})
    return d


def cramers_v(x, y):
    ia, ua = pd.factorize(x)
    ib, ub = pd.factorize(y)
    ra, kb = len(ua), len(ub)
    n = len(ia)
    ct = np.bincount(ia * kb + ib, minlength=ra * kb).reshape(ra, kb).astype(float)
    exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((ct - exp) ** 2 / exp)
    phi2 = max(0.0, chi2 / n - (ra - 1) * (kb - 1) / (n - 1))
    rc = ra - (ra - 1) ** 2 / (n - 1)
    kc = kb - (kb - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan


def corr_ratio(cat, num):
    mask = num.notna().to_numpy()
    codes, _ = pd.factorize(cat[mask])
    x = num[mask].to_numpy(dtype=float)
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=x) / counts
    grand = x.mean()
    ss_between = (counts * (means - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan


real = clean(pd.read_csv(real_path))

num_cols = [
    c for c in real.columns
    if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
]
cat_cols = [
    c for c in real.columns
    if c not in num_cols and real[c].nunique() <= max_categories
]
feats = num_cols + cat_cols

target = None
for c in cat_cols:
    vals = {str(v).lower() for v in real[c].dropna().unique()}
    if {"alive", "dead"} <= vals:
        target = c

tri_cols = [c for c in cat_cols if c != target and real[c].nunique() <= triple_max_levels]


def prep(d):
    d = d.reindex(columns=real.columns).reset_index(drop=True)
    out = pd.DataFrame(index=d.index)
    for c in num_cols:
        out[c] = pd.to_numeric(d[c], errors="coerce")
    for c in cat_cols:
        out[c] = d[c].astype(str)
    return out


def assoc_matrix(d):
    cols = num_cols + cat_cols
    k = len(cols)
    m = np.eye(k)
    spear = d[num_cols].corr(method="spearman").abs()
    for i, j in combinations(range(k), 2):
        a, b = cols[i], cols[j]
        if a in num_cols and b in num_cols:
            v = spear.loc[a, b]
        elif a in cat_cols and b in cat_cols:
            v = cramers_v(d[a], d[b])
        elif a in cat_cols:
            v = corr_ratio(d[a], d[b])
        else:
            v = corr_ratio(d[b], d[a])
        m[i, j] = m[j, i] = v
    return np.nan_to_num(m)


def c2st(r_raw, s_raw, name):
    r_raw = r_raw.reindex(columns=real.columns)
    s_raw = s_raw.reindex(columns=real.columns)
    n = min(len(r_raw), len(s_raw))
    combined = pd.concat(
        [r_raw.sample(n=n, random_state=seed)[feats], s_raw.sample(n=n, random_state=seed)[feats]],
        ignore_index=True,
    )
    x = pd.DataFrame(index=combined.index)
    for c in num_cols:
        x[c] = pd.to_numeric(combined[c], errors="coerce")
    for c in cat_cols:
        x[c] = pd.factorize(combined[c].astype(str), sort=True)[0]
    y = np.r_[np.zeros(n), np.ones(n)]
    mask = [c in cat_cols for c in x.columns]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    clf = HistGradientBoostingClassifier(categorical_features=mask, max_iter=200, random_state=seed)
    clf.fit(x_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])
    pi = permutation_importance(clf, x_te, y_te, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=-1)
    imp = (
        pd.DataFrame({"synthetic": name, "column": x.columns, "auc_drop": pi.importances_mean})
        .sort_values("auc_drop", ascending=False)
        .head(5)
    )
    return auc, imp


mu = real[num_cols].mean()
sd = real[num_cols].std(ddof=1).replace(0, np.nan)
med = real[num_cols].median()


def zscore(d):
    return ((d[num_cols].fillna(med) - mu) / sd).fillna(0).to_numpy()


rng = np.random.default_rng(seed)
dirs = rng.normal(size=(n_proj, len(num_cols)))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

real_p = prep(real)
real_proj = zscore(real_p) @ dirs.T
real_assoc = assoc_matrix(real_p)
off = np.triu_indices(real_assoc.shape[0], 1)

summary_rows, imp_frames, tri_rows = [], [], []

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    auc, imp = c2st(real, syn, name)
    imp_frames.append(imp)

    if len(syn) > syn_sample:
        syn = syn.sample(n=syn_sample, random_state=seed)
    syn_p = prep(syn)

    syn_proj = zscore(syn_p) @ dirs.T
    sliced = np.mean([stats.wasserstein_distance(real_proj[:, i], syn_proj[:, i]) for i in range(n_proj)])

    diff = np.abs(real_assoc - assoc_matrix(syn_p))[off]

    tri_vals = []
    if target is not None:
        codes_r, codes_s, ks = {}, {}, {}
        for c in [target] + tri_cols:
            cats = sorted(set(real_p[c].unique()) | set(syn_p[c].unique()))
            codes_r[c] = pd.Categorical(real_p[c], categories=cats).codes.astype(np.int64)
            codes_s[c] = pd.Categorical(syn_p[c], categories=cats).codes.astype(np.int64)
            ks[c] = len(cats)
        for a, b in combinations(tri_cols, 2):
            size = ks[target] * ks[a] * ks[b]
            jr = (codes_r[target] * ks[a] + codes_r[a]) * ks[b] + codes_r[b]
            js = (codes_s[target] * ks[a] + codes_s[a]) * ks[b] + codes_s[b]
            pr = np.bincount(jr, minlength=size) / len(jr)
            ps = np.bincount(js, minlength=size) / len(js)
            tvd = 0.5 * np.abs(pr - ps).sum()
            tri_vals.append(tvd)
            tri_rows.append({"synthetic": name, "target": target, "col_a": a, "col_b": b, "joint_tvd": tvd})

    summary_rows.append({
        "synthetic": name,
        "c2st_auc": auc,
        "sliced_wasserstein": sliced,
        "assoc_rms_diff": np.sqrt(np.mean(diff ** 2)),
        "assoc_max_diff": diff.max(),
        "triple_tvd_mean": np.mean(tri_vals) if tri_vals else np.nan,
    })

summary = pd.DataFrame(summary_rows).set_index("synthetic")
summary["c2st_dev"] = (summary["c2st_auc"] - 0.5).abs()
rank_cols = ["c2st_dev", "sliced_wasserstein", "assoc_rms_diff"]
if summary["triple_tvd_mean"].notna().all():
    rank_cols.append("triple_tvd_mean")
summary["rank"] = summary[rank_cols].rank().mean(axis=1).rank()
print(summary.drop(columns="c2st_dev").sort_values("rank").round(4))

print(pd.concat(imp_frames).round(4).to_string(index=False))

if tri_rows:
    tri_res = pd.DataFrame(tri_rows)
    print(
        tri_res.sort_values("joint_tvd", ascending=False)
        .groupby("synthetic")
        .head(10)
        .sort_values(["synthetic", "joint_tvd"], ascending=[True, False])
        .round(4)
        .to_string(index=False)
    )
