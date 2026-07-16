# -*- coding: utf-8 -*-
"""
Bootstrap CV analysis (one-sided)
Hypothesis: CV(ZD) > CV(control)
"""

import numpy as np
import pandas as pd

# -------- SETTINGS --------
input_file = r"C:/Users/homay/OneDrive - UHN/Desktop/Slicecultureprismfiles/merged_csv_files.xlsx"
output_file = r"C:/Users/homay/OneDrive - UHN/Desktop/bootstrapsliceculture_results.xlsx"
n_boot = 10000


# -------- FUNCTIONS --------
def compute_cv(x):
    x = pd.Series(x)
    x = pd.to_numeric(x, errors='coerce')
    x = x.dropna()

    if len(x) < 2:
        return np.nan

    mean_val = np.mean(x)
    if mean_val == 0:
        return np.nan

    # 🔥 KEY FIX: use absolute mean
    return np.std(x, ddof=1) / np.abs(mean_val)

def bootstrap_cv_diff(control, zd, n_boot=10000, alternative="greater", seed=1234):
    rng = np.random.default_rng(seed)

    cv_control = compute_cv(control)
    cv_zd = compute_cv(zd)

    actual_diff = cv_zd - cv_control

    diffs = []

    for _ in range(n_boot):
        s_control = rng.choice(
            control,
            size=len(control),
            replace=True
        )

        s_zd = rng.choice(
            zd,
            size=len(zd),
            replace=True
        )

        diff = compute_cv(s_zd) - compute_cv(s_control)

        if np.isfinite(diff):
            diffs.append(diff)

    diffs = np.asarray(diffs)

    # -----------------------------------------
    # Two-sided 95% CI for the CV difference
    # -----------------------------------------

    lower, upper = np.percentile(
        diffs,
        [2.5, 97.5]
    )

    # -----------------------------------------
    # Approximate null distribution
    # -----------------------------------------
    # Bootstrap diffs are centred around actual_diff.
    # Subtract actual_diff to centre them around H0 = 0.

    null_diffs = diffs - actual_diff

    # -----------------------------------------
    # Bootstrap p-value
    # -----------------------------------------

    if alternative == "greater":
        # H1: CV_ZD > CV_control
        p_value = (
            np.sum(null_diffs >= actual_diff) + 1
        ) / (len(null_diffs) + 1)

    elif alternative == "two-sided":
        # H1: CV_ZD != CV_control
        p_value = (
            np.sum(
                np.abs(null_diffs)
                >= np.abs(actual_diff)
            ) + 1
        ) / (len(null_diffs) + 1)

    else:
        raise ValueError(
            "alternative must be 'greater' or 'two-sided'"
        )

    return lower, upper, p_value, diffs


# -------- LOAD FILE --------
xls = pd.ExcelFile(input_file)

results = []
all_bootstrap = []


# -------- LOOP THROUGH SHEETS --------
for sheet in xls.sheet_names:
    df = xls.parse(sheet)
    df = df.dropna(axis=1, how='all')

    if df.shape[1] < 2:
        print(f"Skipping {sheet} (not enough columns)")
        continue

    # Column 0 = control, Column 1 = ZD
    control = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    zd = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values

    if len(control) < 2 or len(zd) < 2:
        print(f"Skipping {sheet} (not enough valid data)")
        continue

    cv_control = compute_cv(control)
    cv_zd = compute_cv(zd)

    lower, upper, p_value, diffs = bootstrap_cv_diff(control, zd, n_boot, alternative="two-sided")

    mean_diff = np.mean(diffs)

    results.append({
        "Sheet": sheet,
        "CV_control": cv_control,
        "CV_ZD": cv_zd,
        "Diff_CV (ZD - control)": cv_zd - cv_control,
        "Bootstrap_mean_diff": mean_diff,
        "CI_lower (5%)": lower,
        "CI_upper": upper,
        "p_value (one-sided)": p_value,
        "Significant": "Yes" if p_value < 0.05 else "No"
    })

    temp_df = pd.DataFrame({
        "Sheet": sheet,
        "Bootstrap_Diff": diffs
    })
    all_bootstrap.append(temp_df)


# -------- SAVE --------
results_df = pd.DataFrame(results)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Summary", index=False)

    if all_bootstrap:
        bootstrap_df = pd.concat(all_bootstrap, ignore_index=True)
        bootstrap_df.to_excel(writer, sheet_name="Bootstrap_Distribution", index=False)

print("Results saved to:")
print(output_file)