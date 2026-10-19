import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

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

qs = [0.05, 0.25, 0.5, 0.75, 0.95]


def clean(d):
    d = d.drop(columns=drop_cols, errors="ignore")
    if "Sex" in d.columns:
        d["Sex"] = d["Sex"].replace({"MALE": "Male", "FEMALE": "Female"})
    return d


real = clean(pd.read_csv(real_path))
rows = []

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    for col in real.columns:
        if col not in syn.columns:
            continue
        r_all, s_all = real[col], syn[col]
        base = {
            "synthetic": name,
            "column": col,
            "real_missing_rate": r_all.isna().mean(),
            "syn_missing_rate": s_all.isna().mean(),
        }

        if pd.api.types.is_numeric_dtype(r_all) and not pd.api.types.is_bool_dtype(r_all):
            r = r_all.dropna().astype(float).to_numpy()
            s = pd.to_numeric(s_all, errors="coerce").dropna().to_numpy()
            r_std = np.std(r, ddof=1)
            scale = r_std if r_std > 0 else np.nan
            rq = np.quantile(r, qs)
            sq = np.quantile(s, qs)
            r_var = np.var(r, ddof=1)
            s_var = np.var(s, ddof=1)
            w = stats.wasserstein_distance(r, s)
            row = {
                **base,
                "type": "numeric",
                "real_mean": r.mean(),
                "syn_mean": s.mean(),
                "mean_diff_std": abs(r.mean() - s.mean()) / scale,
                "real_median": np.median(r),
                "syn_median": np.median(s),
                "median_diff_std": abs(np.median(r) - np.median(s)) / scale,
                "real_var": r_var,
                "syn_var": s_var,
                "var_ratio": s_var / r_var if r_var > 0 else np.nan,
                "quantile_mae_std": np.mean(np.abs(rq - sq)) / scale,
                "ks_statistic": stats.ks_2samp(r, s).statistic,
                "wasserstein": w,
                "wasserstein_norm": w / scale,
            }
        else:
            r = r_all.dropna().astype(str)
            s = s_all.dropna().astype(str)
            rp = r.value_counts(normalize=True)
            sp = s.value_counts(normalize=True)
            cats = rp.index.union(sp.index)
            rp = rp.reindex(cats, fill_value=0).to_numpy()
            sp = sp.reindex(cats, fill_value=0).to_numpy()
            row = {
                **base,
                "type": "categorical",
                "tvd": 0.5 * np.abs(rp - sp).sum(),
                "js_distance": jensenshannon(rp, sp, base=2),
                "category_coverage": len(set(r.unique()) & set(s.unique())) / r.nunique(),
            }
        rows.append(row)

report = pd.DataFrame(rows)

numeric_cols = [
    "synthetic", "column", "real_mean", "syn_mean", "mean_diff_std",
    "real_median", "syn_median", "median_diff_std", "real_var", "syn_var",
    "var_ratio", "quantile_mae_std", "ks_statistic", "wasserstein_norm",
    "real_missing_rate", "syn_missing_rate",
]
categorical_cols = [
    "synthetic", "column", "tvd", "js_distance", "category_coverage",
    "real_missing_rate", "syn_missing_rate",
]

numeric_report = report[report["type"] == "numeric"][numeric_cols]
categorical_report = report[report["type"] == "categorical"][categorical_cols]

print(numeric_report.sort_values(["synthetic", "ks_statistic"], ascending=[True, False]).round(4).to_string(index=False))
print(categorical_report.sort_values(["synthetic", "tvd"], ascending=[True, False]).round(4).to_string(index=False))

summary = report.groupby("synthetic").agg(
    mean_ks=("ks_statistic", "mean"),
    mean_wasserstein_norm=("wasserstein_norm", "mean"),
    mean_quantile_mae_std=("quantile_mae_std", "mean"),
    mean_tvd=("tvd", "mean"),
    mean_js=("js_distance", "mean"),
    mean_coverage=("category_coverage", "mean"),
)
summary["rank"] = (
    summary[["mean_ks", "mean_wasserstein_norm", "mean_tvd"]]
    .rank()
    .mean(axis=1)
    .rank()
)

print(summary.sort_values("rank").round(4))
