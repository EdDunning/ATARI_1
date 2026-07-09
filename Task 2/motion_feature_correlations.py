"""
===========================================================================
Motion Feature Correlation Analysis
===========================================================================

Purpose
-------
This script explores the relationship between the extracted motion features
and the expert OSATS ratings provided in the JIGSAWS metadata. It performs
correlation analysis to identify which motion features are most strongly
associated with surgical skill and therefore are likely to be informative
inputs to the machine learning classifier developed in the next stage of the
project.

How it works
------------
1. Loads the merged datasets for the Suturing, Needle Passing and Knot Tying
   tasks.
2. Identifies all numerical motion feature columns.
3. Calculates the correlation between every motion feature and:
      - Global Rating Score (GRS)
      - Respect for Tissue
      - Suture/Needle Handling
      - Time and Motion
      - Flow of Operation
      - Overall Performance
      - Quality of Final Product
4. Produces correlation tables for:
      - all tasks combined
      - each task individually
5. Saves the results as CSV files for further analysis.

Correlation Measures
--------------------
Two correlation coefficients are calculated:

Spearman's Rank Correlation (ρ)
    Measures the strength of a monotonic relationship between two variables
    using their ranked values. It does not assume a linear relationship and
    is appropriate for ordinal data such as the OSATS scores (1–5). This is
    the primary metric used in this project.

Pearson's Correlation Coefficient (r)
    Measures the strength of a linear relationship between two continuous
    variables. Although widely used, it assumes interval-scale data and
    linear relationships, making it less appropriate for the ordinal OSATS
    ratings. It is included here for comparison only.

Interpretation
--------------
Correlation coefficients lie between -1 and +1.

    +1   Perfect positive relationship
     0   No relationship
    -1   Perfect negative relationship

Features with larger absolute Spearman correlation coefficients are considered
to have a stronger association with surgical skill and are therefore likely to
be more useful predictors during machine learning model development.
"""


from pathlib import Path

import pandas as pd


TASK2_DIR = Path(__file__).resolve().parent

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

TARGET_COLUMNS = [
    "grs",
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]

META_COLUMNS = {
    "file",
    "filename",
    "experience_level",
    "grs",
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
    "task",
}

OUTPUT_LONG_ALL = TASK2_DIR / "motion_feature_correlations_all_tasks.csv"
OUTPUT_LONG_BY_TASK = TASK2_DIR / "motion_feature_correlations_by_task.csv"
OUTPUT_WIDE_ALL = TASK2_DIR / "motion_feature_correlations_matrix_all_tasks.csv"


def load_merged_datasets() -> pd.DataFrame:
    frames = []

    for task_name, file_path in MERGED_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Merged file not found: {file_path}")

        df = pd.read_csv(file_path)
        df["task"] = task_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []

    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def compute_correlations(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    rows = []

    for feature in feature_cols:
        for target in TARGET_COLUMNS:
            if target not in df.columns:
                continue

            pair = df[[feature, target]].dropna()
            n = len(pair)

            if n < 2:
                continue

            spearman_rho = pair[feature].corr(pair[target], method="spearman")
            pearson_r = pair[feature].corr(pair[target], method="pearson")

            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "spearman_rho": spearman_rho,
                    "pearson_r": pearson_r,
                    "abs_spearman_rho": abs(spearman_rho) if pd.notna(spearman_rho) else None,
                    "abs_pearson_r": abs(pearson_r) if pd.notna(pearson_r) else None,
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=["target", "abs_spearman_rho", "feature"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    return result


def main() -> None:
    combined = load_merged_datasets()

    # Overall correlations using all three tasks together.
    overall = compute_correlations(combined, group_name="all_tasks")
    overall.to_csv(OUTPUT_LONG_ALL, index=False)

    # Wide matrix version for easier viewing in spreadsheets.
    wide = overall.pivot(index="feature", columns="target", values="spearman_rho")
    wide.to_csv(OUTPUT_WIDE_ALL)

    # Separate correlations for each task.
    by_task_frames = []
    for task_name, task_df in combined.groupby("task", sort=False):
        task_corr = compute_correlations(task_df, group_name=task_name)
        by_task_frames.append(task_corr)

    by_task = pd.concat(by_task_frames, ignore_index=True) if by_task_frames else pd.DataFrame()
    by_task.to_csv(OUTPUT_LONG_BY_TASK, index=False)

    print(f"Saved overall correlations to: {OUTPUT_LONG_ALL}")
    print(f"Saved overall Spearman matrix to: {OUTPUT_WIDE_ALL}")
    print(f"Saved task-specific correlations to: {OUTPUT_LONG_BY_TASK}")

    if not overall.empty:
        print("\nTop correlations by absolute Spearman rho:")
        print(overall.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

