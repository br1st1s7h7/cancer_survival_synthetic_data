import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)

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
min_support = 200
purity = 0.99
min_expected = 30
min_rows = 50
n_bins = 5
band_min_unique = 20
overlap_tol = 0.01
broken_tol = 0.05
syn_sample = 300000


def clean(d):
    d = d.drop(columns=drop_cols, errors="ignore")
    if "Sex" in d.columns:
        d["Sex"] = d["Sex"].replace({"MALE": "Male", "FEMALE": "Female"})
    return d


def disc_str(s):
    return pd.to_numeric(s, errors="coerce").astype(float).round(6).astype(str)


real = clean(pd.read_csv(real_path))

num_cols = [
    c for c in real.columns
    if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
]
cat_cols = [
    c for c in real.columns
    if c not in num_cols and real[c].nunique() <= max_categories
]
disc_num = [c for c in num_cols if real[c].nunique() <= 30]
disc_cols = cat_cols + disc_num

target = None
for c in cat_cols:
    vals = {str(v).lower() for v in real[c].dropna().unique()}
    if {"alive", "dead"} <= vals:
        target = c


def prep(d):
    d = d.reindex(columns=real.columns).reset_index(drop=True)
    n = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") for c in num_cols})
    s = pd.DataFrame({c: (disc_str(d[c]) if c in disc_num else d[c].astype(str)) for c in disc_cols})
    return n, s


RN, RD = prep(real)
RV = {c: sorted(RD[c].unique()) for c in disc_cols}
K = {c: len(RV[c]) for c in disc_cols}


def codes(D, c):
    return pd.Categorical(D[c], categories=RV[c]).codes.astype(np.int64)


def joint(ca, cb, ka, kb):
    m = ca >= 0
    return np.bincount(ca[m] * (kb + 1) + cb[m] + 1, minlength=ka * (kb + 1)).reshape(ka, kb + 1).astype(float)


RC = {c: codes(RD, c) for c in disc_cols}

rules, forb = [], {}
for a_col in cat_cols:
    for b_col in disc_cols:
        if a_col == b_col or K[b_col] < 2:
            continue
        ct = joint(RC[a_col], RC[b_col], K[a_col], K[b_col])[:, 1:]
        n = ct.sum()
        rs = ct.sum(axis=1)
        cs = ct.sum(axis=0) / n
        for ai in np.where(rs >= min_support)[0]:
            bi = ct[ai].argmax()
            pur = ct[ai, bi] / rs[ai]
            if pur >= purity and cs[bi] < 0.97:
                rules.append((a_col, ai, b_col, bi, rs[ai], pur))
        if b_col in cat_cols and cat_cols.index(a_col) < cat_cols.index(b_col):
            exp = rs[:, None] * cs[None, :]
            mask = (ct == 0) & (exp >= min_expected)
            if mask.any():
                forb[(a_col, b_col)] = mask

bands = []
cand_num = [c for c in num_cols if real[c].nunique() >= band_min_unique]
for c_col in cat_cols:
    if K[c_col] < 2:
        continue
    for n_col in cand_num:
        df = pd.DataFrame({"c": RD[c_col], "v": RN[n_col]}).dropna()
        g = df.groupby("c")["v"].agg(["min", "max", "size"])
        g = g[g["size"] >= min_support]
        if len(g) < 2:
            continue
        v = df["v"].to_numpy()
        inside = ((v[:, None] >= g["min"].to_numpy()[None, :]) & (v[:, None] <= g["max"].to_numpy()[None, :])).sum(axis=1)
        if (inside > 1).mean() <= overlap_tol:
            bands.append((c_col, n_col, g["min"].to_dict(), g["max"].to_dict()))

edges = {}
for n_col in num_cols:
    if real[n_col].nunique() > 10:
        q = np.unique(np.quantile(RN[n_col].dropna(), np.linspace(0, 1, n_bins + 1)))
        e = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
        if len(e) >= 3:
            edges[n_col] = e


def level_labels(N, D):
    out = {}
    for c in cat_cols:
        if c != target:
            out[c] = D[c].to_numpy()
    for n_col, e in edges.items():
        out[n_col + "_bin"] = pd.cut(N[n_col], bins=e, labels=False).fillna(-1).astype(int).astype(str).to_numpy()
    return out


def survival_rows(name, rl, sl, ra, sa):
    rows = []
    for c in rl:
        rg = pd.DataFrame({"a": ra, "l": rl[c]}).groupby("l")["a"].agg(["mean", "size"])
        sg = pd.DataFrame({"a": sa, "l": sl[c]}).groupby("l")["a"].agg(["mean", "size"])
        rg = rg[rg["size"] >= min_rows]
        common = rg.index.intersection(sg.index[sg["size"] >= min_rows])
        if len(common) < 2:
            continue
        rm = rg.loc[common, "mean"]
        sm = sg.loc[common, "mean"]
        rw = rg.loc[common, "size"]
        sp = stats.spearmanr(rm, sm)[0] if len(common) >= 3 else np.nan
        rows.append({
            "synthetic": name,
            "column": c,
            "real_spread": rm.max() - rm.min(),
            "syn_spread": sm.max() - sm.min(),
            "gap": np.average(np.abs(rm - sm), weights=rw),
            "spearman": sp,
            "levels_lost_share": 1 - rw.sum() / rg["size"].sum(),
        })
    return rows


if target is not None:
    real_alive = (RD[target].str.lower() == "alive").to_numpy()
    real_labels = level_labels(RN, RD)

range_rows, rule_rows, forb_rows, band_rows, surv_rows = [], [], [], [], []

for name, path in synthetic_paths.items():
    syn = clean(pd.read_csv(path))
    if syn_sample and len(syn) > syn_sample:
        syn = syn.sample(n=syn_sample, random_state=0)
    SN, SD = prep(syn)
    SC = {c: codes(SD, c) for c in disc_cols}

    for n_col in num_cols:
        r = RN[n_col].dropna()
        s = SN[n_col].dropna()
        nonint = np.nan
        if np.allclose(r, np.round(r)):
            nonint = (np.abs(s - np.round(s)) > 1e-6).mean()
        range_rows.append({
            "synthetic": name,
            "column": n_col,
            "out_of_range": ((s < r.min()) | (s > r.max())).mean(),
            "non_integer": nonint,
        })

    cache = {}
    for a_col, ai, b_col, bi, sup, pur in rules:
        key = (a_col, b_col)
        if key not in cache:
            cache[key] = joint(SC[a_col], SC[b_col], K[a_col], K[b_col])
        row = cache[key][ai]
        ssup = row.sum()
        viol = 1 - row[bi + 1] / ssup if ssup > 0 else 1.0
        rule_rows.append({
            "synthetic": name,
            "rule": f"{a_col}={RV[a_col][ai]} -> {b_col}={RV[b_col][bi]}",
            "real_support": sup,
            "real_purity": pur,
            "syn_support": ssup,
            "syn_violation": viol,
        })

    for (a_col, b_col), mask in forb.items():
        ct = joint(SC[a_col], SC[b_col], K[a_col], K[b_col])[:, 1:]
        forb_rows.append({
            "synthetic": name,
            "col_a": a_col,
            "col_b": b_col,
            "forbidden_cells": int(mask.sum()),
            "syn_row_share": ct[mask].sum() / len(SD),
        })

    for c_col, n_col, lo, hi in bands:
        s = pd.DataFrame({"c": SD[c_col], "v": SN[n_col]}).dropna()
        lo_v = s["c"].map(lo)
        hi_v = s["c"].map(hi)
        m = lo_v.notna()
        if m.sum() == 0:
            continue
        out = ((s["v"][m] < lo_v[m]) | (s["v"][m] > hi_v[m])).mean()
        band_rows.append({
            "synthetic": name,
            "category": c_col,
            "numeric": n_col,
            "syn_violation": out,
        })

    if target is not None:
        syn_alive = (SD[target].str.lower() == "alive").to_numpy()
        surv_rows += survival_rows(name, real_labels, level_labels(SN, SD), real_alive, syn_alive)

range_df = pd.DataFrame(range_rows, columns=["synthetic", "column", "out_of_range", "non_integer"])
rule_df = pd.DataFrame(rule_rows, columns=["synthetic", "rule", "real_support", "real_purity", "syn_support", "syn_violation"])
forb_df = pd.DataFrame(forb_rows, columns=["synthetic", "col_a", "col_b", "forbidden_cells", "syn_row_share"])
band_df = pd.DataFrame(band_rows, columns=["synthetic", "category", "numeric", "syn_violation"])
surv_df = pd.DataFrame(surv_rows, columns=["synthetic", "column", "real_spread", "syn_spread", "gap", "spearman", "levels_lost_share"])


def pick(df, col, name):
    s = df[df["synthetic"] == name][col]
    return s.mean() if len(s) else np.nan


summary_rows = []
for name in synthetic_paths:
    rr = rule_df[rule_df["synthetic"] == name]
    summary_rows.append({
        "synthetic": name,
        "out_of_range": pick(range_df, "out_of_range", name),
        "non_integer": pick(range_df, "non_integer", name),
        "band_violation": pick(band_df, "syn_violation", name),
        "rule_violation": np.average(rr["syn_violation"], weights=rr["real_support"]) if len(rr) else np.nan,
        "rules_broken_share": (rr["syn_violation"] > broken_tol).mean() if len(rr) else np.nan,
        "forbidden_share": pick(forb_df, "syn_row_share", name),
        "survival_gap": pick(surv_df, "gap", name),
        "survival_spearman": pick(surv_df, "spearman", name),
    })

summary = pd.DataFrame(summary_rows).set_index("synthetic")
low_better = ["out_of_range", "non_integer", "band_violation", "rule_violation", "forbidden_share", "survival_gap"]
ranks = pd.concat(
    [summary[low_better].rank(), summary[["survival_spearman"]].rank(ascending=False)],
    axis=1,
)
summary["rank"] = ranks.mean(axis=1).rank()

print(len(rules), len(forb), len(bands))
print(summary.sort_values("rank").round(4))

print(
    rule_df.sort_values("syn_violation", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "syn_violation"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)
print(
    forb_df.sort_values("syn_row_share", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "syn_row_share"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)
print(
    band_df.sort_values("syn_violation", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "syn_violation"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)
print(
    surv_df[surv_df["real_spread"] >= 0.05]
    .sort_values("gap", ascending=False)
    .groupby("synthetic")
    .head(10)
    .sort_values(["synthetic", "gap"], ascending=[True, False])
    .round(4)
    .to_string(index=False)
)
