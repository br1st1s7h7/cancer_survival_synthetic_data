import numpy as np
import pandas as pd
from scipy import stats

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
n_bins = 5
min_rows = 30
syn_sample = 300000


def clean(d):
    d = d.drop(columns=drop_cols, errors="ignore")
    if "Sex" in d.columns:
        d["Sex"] = d["Sex"].replace({"MALE": "Male", "FEMALE": "Female"})
    return d


def base_name(c):
    return c[:-4] if c.endswith("_bin") else c


real = clean(pd.read_csv(real_path))

num_cols = [
    c for c in real.columns
    if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
]
cat_cols = [
    c for c in real.columns
    if c not in num_cols and real[c].nunique() <= max_categories
]

edges = {}
for c in num_cols:
    if real[c].nunique() > 10:
        q = np.unique(np.quantile(real[c].dropna(), np.linspace(0, 1, n_bins + 1)))
        e = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
        if len(e) >= 3:
            edges[c] = e

all_cats = cat_cols + [c + "_bin" for c in edges]

target = None
for c in cat_cols:
    vals = {str(v).lower() for v in real[c].dropna().unique()}
    if {"alive", "dead"} <= vals:
        target = c

real_miss = real.isna().mean()
real_levels = {c: real[c].astype(str).value_counts(normalize=True) for c in cat_cols}


def prep(d):
    d = d.reset_index(drop=True)
    out = pd.DataFrame(index=d.index)
    for c in num_cols:
        out[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    for c in cat_cols:
        out[c] = d[c].astype(str) if c in d.columns else "nan"
    for c, e in edges.items():
        out[c + "_bin"] = pd.cut(out[c], bins=e, labels=False).fillna(-1).astype(int).astype(str)
    return out


def cond_numeric(r, s, name, scale):
    rows = []
    for cond in all_cats:
        rg = r.groupby(cond, sort=False).indices
        sg = s.groupby(cond, sort=False).indices
        for n in num_cols:
            if base_name(cond) == n or not np.isfinite(scale[n]) or scale[n] <= 0:
                continue
            rn = r[n].to_numpy()
            sn = s[n].to_numpy()
            ks_list, w_list, wt_list = [], [], []
            uncovered, total = 0.0, 0.0
            for level, ri in rg.items():
                rv = rn[ri]
                rv = rv[~np.isnan(rv)]
                if len(rv) < min_rows:
                    continue
                total += len(rv)
                si = sg.get(level)
                sv = sn[si] if si is not None else np.array([])
                sv = sv[~np.isnan(sv)]
                if len(sv) < min_rows:
                    uncovered += len(rv)
                    continue
                ks_list.append(stats.ks_2samp(rv, sv).statistic)
                w_list.append(stats.wasserstein_distance(rv, sv) / scale[n])
                wt_list.append(len(rv))
            if not wt_list:
                continue
            rows.append({
                "synthetic": name,
                "given": cond,
                "column": n,
                "weighted_ks": np.average(ks_list, weights=wt_list),
                "weighted_wass_norm": np.average(w_list, weights=wt_list),
                "uncovered_share": uncovered / total if total > 0 else np.nan,
            })
    return rows


def cond_categorical(r, s, name):
    rows = []
    codes_r, codes_s, ks = {}, {}, {}
    for c in all_cats:
        cats = sorted(set(r[c].unique()) | set(s[c].unique()))
        codes_r[c] = pd.Categorical(r[c], categories=cats).codes.astype(np.int64)
        codes_s[c] = pd.Categorical(s[c], categories=cats).codes.astype(np.int64)
        ks[c] = len(cats)
    for a in all_cats:
        for b in all_cats:
            if base_name(a) == base_name(b):
                continue
            ka, kb = ks[a], ks[b]
            rc = np.bincount(codes_r[a] * kb + codes_r[b], minlength=ka * kb).reshape(ka, kb).astype(float)
            sc = np.bincount(codes_s[a] * kb + codes_s[b], minlength=ka * kb).reshape(ka, kb).astype(float)
            rn = rc.sum(axis=1)
            sn = sc.sum(axis=1)
            valid = rn >= min_rows
            if not valid.any():
                continue
            rp = rc[valid] / rn[valid][:, None]
            sp = np.divide(sc[valid], sn[valid][:, None], out=np.zeros_like(sc[valid]), where=sn[valid][:, None] > 0)
            tvd = 0.5 * np.abs(rp - sp).sum(axis=1)
            tvd[sn[valid] == 0] = 1.0
            rows.append({
                "synthetic": name,
                "given": a,
                "column": b,
                "weighted_tvd": np.average(tvd, weights=rn[valid]),
            })
    return rows


real_prep = prep(real)
scale = {n: np.nanstd(real_prep[n].to_numpy(), ddof=1) for n in num_cols}

num_rows, cat_rows, miss_rows, lost_rows = [], [], [], []

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    if syn_sample and len(syn) > syn_sample:
        syn = syn.sample(n=syn_sample, random_state=0)
    for c in real.columns:
        if c not in syn.columns:
            continue
        miss_rows.append({
            "synthetic": name,
            "column": c,
            "real_missing": real_miss[c],
            "syn_missing": syn[c].isna().mean(),
        })
        if c in cat_cols:
            sl = syn[c].astype(str).value_counts(normalize=True)
            for level, share in real_levels[c].items():
                if share > 0.005 and sl.get(level, 0) == 0:
                    lost_rows.append({
                        "synthetic": name,
                        "column": c,
                        "level": level,
                        "real_share": share,
                    })
    syn_prep = prep(syn)
    num_rows += cond_numeric(real_prep, syn_prep, name, scale)
    cat_rows += cond_categorical(real_prep, syn_prep, name)

num_res = pd.DataFrame(num_rows)
cat_res = pd.DataFrame(cat_rows)
miss_res = pd.DataFrame(miss_rows)
miss_res = miss_res[(miss_res["real_missing"] - miss_res["syn_missing"]).abs() > 0.01]
lost_res = pd.DataFrame(lost_rows)

summary = pd.concat(
    [
        num_res.groupby("synthetic")[["weighted_ks", "weighted_wass_norm", "uncovered_share"]].mean(),
        cat_res.groupby("synthetic")[["weighted_tvd"]].mean(),
    ],
    axis=1,
)
summary["rank"] = (
    summary[["weighted_ks", "weighted_wass_norm", "weighted_tvd"]]
    .rank()
    .mean(axis=1)
    .rank()
)
print(summary.sort_values("rank").round(4))

print(miss_res.round(4).to_string(index=False))
print(lost_res.round(4).to_string(index=False))

print(
    num_res.sort_values("weighted_ks", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "weighted_ks"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)
print(
    cat_res.sort_values("weighted_tvd", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "weighted_tvd"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)

if target is not None:
    print(
        cat_res[cat_res["column"] == target]
        .sort_values(["synthetic", "weighted_tvd"], ascending=[True, False])
        .round(4)
        .to_string(index=False)
    )
    print(
        num_res[num_res["given"] == target]
        .sort_values(["synthetic", "weighted_ks"], ascending=[True, False])
        .round(4)
        .to_string(index=False)
    )
