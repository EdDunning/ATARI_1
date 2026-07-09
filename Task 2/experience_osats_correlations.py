'''
This script computes the correlation between experience level and OSATS scores across all tasks and for 
each task separately. It saves the results in both long and wide formats for further analysis.
'''
from pathlib import Path

import pandas as pd


TASK2_DIR = Path(__file__).resolve().parent

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

# Include grs as a reference target. Remove it if you only want the six OSATS scores.
TARGET_COLUMNS = [
    "grs",
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]

EXPERIENCE_MAP = {
    "N": 1, "Novice": 1,
    "I": 2, "Intermediate": 2,
    "E": 3, "Expert": 3,
}

META_COLUMNS = {
    "file",
    "filename",
    "task",
    "experience_level",
    "experience_level_numeric",
    "grs",
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
}

OUTPUT_LONG_ALL = TASK2_DIR / "experience_osats_correlations_all_tasks.csv"
OUTPUT_LONG_BY_TASK = TASK2_DIR / "experience_osats_correlations_by_task.csv"
OUTPUT_WIDE_ALL = TASK2_DIR / "experience_osats_correlation_matrix_all_tasks.csv"


def load_merged_datasets() -> pd.DataFrame:
    frames = []

    for task_name, file_path in MERGED_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing merged file: {file_path}")

        df = pd.read_csv(file_path)
        df["task"] = task_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def encode_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert experience labels to an ordinal numeric scale:
    Novice = 1, Intermediate = 2, Expert = 3.
    """
    if "experience_level" not in df.columns:
        raise ValueError("Expected an 'experience_level' column in the merged data.")

    out = df.copy()

    if pd.api.types.is_numeric_dtype(out["experience_level"]):
        out["experience_level_numeric"] = pd.to_numeric(out["experience_level"], errors="coerce")
    else:
        out["experience_level_numeric"] = out["experience_level"].map(EXPERIENCE_MAP)

    if out["experience_level_numeric"].isna().any():
        bad_values = sorted(out.loc[out["experience_level_numeric"].isna(), "experience_level"].astype(str).unique())
        raise ValueError(f"Unrecognised experience labels found: {bad_values}")

    return out


def compute_correlations(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue

        pair = df[["experience_level_numeric", target]].dropna()
        n = len(pair)

        if n < 2:
            continue

        spearman_rho = pair["experience_level_numeric"].corr(pair[target], method="spearman")
        pearson_r = pair["experience_level_numeric"].corr(pair[target], method="pearson")

        rows.append(
            {
                "group": group_name,
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
            by="abs_spearman_rho",
            ascending=False,
        ).reset_index(drop=True)

    return result


def main() -> None:
    combined = load_merged_datasets()
    combined = encode_experience(combined)

    overall = compute_correlations(combined, group_name="all_tasks")
    overall.to_csv(OUTPUT_LONG_ALL, index=False)

    wide = overall.set_index("target")[["spearman_rho", "pearson_r"]]
    wide.to_csv(OUTPUT_WIDE_ALL)

    by_task_frames = []
    for task_name, task_df in combined.groupby("task", sort=False):
        task_corr = compute_correlations(task_df, group_name=task_name)
        by_task_frames.append(task_corr)

    by_task = pd.concat(by_task_frames, ignore_index=True) if by_task_frames else pd.DataFrame()
    by_task.to_csv(OUTPUT_LONG_BY_TASK, index=False)

    print(f"Saved overall correlations to: {OUTPUT_LONG_ALL}")
    print(f"Saved overall correlation matrix to: {OUTPUT_WIDE_ALL}")
    print(f"Saved task-specific correlations to: {OUTPUT_LONG_BY_TASK}")

    if not overall.empty:
        print("\nOverall correlations:")
        print(overall[["target", "n", "spearman_rho", "pearson_r"]].to_string(index=False))


if __name__ == "__main__":
    main()

