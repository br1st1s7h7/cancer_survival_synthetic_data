import numpy as np
import pandas as pd
from itertools import combinations

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
    chi2 = ((ct - exp) ** 2 / exp).sum()
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


def pair_table(d, num_cols, cat_cols):
    out = []
    pear = d[num_cols].corr(method="pearson")
    spear = d[num_cols].corr(method="spearman")
    for a, b in combinations(num_cols, 2):
        out.append(("pearson", a, b, pear.loc[a, b]))
        out.append(("spearman", a, b, spear.loc[a, b]))
    for a, b in combinations(cat_cols, 2):
        out.append(("cramers_v", a, b, cramers_v(d[a], d[b])))
    for c in cat_cols:
        for n in num_cols:
            out.append(("corr_ratio", c, n, corr_ratio(d[c], d[n])))
    return pd.DataFrame(out, columns=["measure", "col_a", "col_b", "value"])


real = clean(pd.read_csv(real_path))

num_cols = [
    c for c in real.columns
    if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
]
cat_cols = [
    c for c in real.columns
    if c not in num_cols and real[c].nunique() <= max_categories
]

real_prep = real[num_cols + cat_cols].copy()
for c in cat_cols:
    real_prep[c] = real_prep[c].astype(str)

real_table = pair_table(real_prep, num_cols, cat_cols)

all_results = []

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    keep = [c for c in num_cols + cat_cols if c in syn.columns]
    syn_prep = syn[keep].copy()
    for c in num_cols:
        if c in syn_prep.columns:
            syn_prep[c] = pd.to_numeric(syn_prep[c], errors="coerce")
    for c in cat_cols:
        if c in syn_prep.columns:
            syn_prep[c] = syn_prep[c].astype(str)
    s_num = [c for c in num_cols if c in syn_prep.columns]
    s_cat = [c for c in cat_cols if c in syn_prep.columns]
    syn_table = pair_table(syn_prep, s_num, s_cat)
    merged = real_table.merge(
        syn_table, on=["measure", "col_a", "col_b"], suffixes=("_real", "_syn")
    )
    merged["abs_diff"] = (merged["value_real"] - merged["value_syn"]).abs()
    merged["synthetic"] = name
    all_results.append(merged)

results = pd.concat(all_results, ignore_index=True)

summary = results.groupby(["synthetic", "measure"])["abs_diff"].mean().unstack()
summary = summary.add_prefix("mean_diff_")
summary["max_diff_overall"] = results.groupby("synthetic")["abs_diff"].max()
summary["rank"] = (
    summary[[c for c in summary.columns if c.startswith("mean_diff_")]]
    .rank()
    .mean(axis=1)
    .rank()
)

print(summary.sort_values("rank").round(4))

for name in synthetic_paths:
    worst = (
        results[results["synthetic"] == name]
        .sort_values("abs_diff", ascending=False)
        .head(10)[["measure", "col_a", "col_b", "value_real", "value_syn", "abs_diff"]]
    )
    print(name)
    print(worst.round(4).to_string(index=False))
