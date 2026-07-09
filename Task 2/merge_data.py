'''
This file merges the motion-feature summary CSVs produced by motion_feature_storage.py with 
the corresponding JIGSAWS metadata tables, producing a single CSV for each task.

'''

from pathlib import Path

import pandas as pd


TASK2_DIR = Path(__file__).resolve().parent
TASK1_DIR = TASK2_DIR.parent / "Task 1"

SUMMARY_FILES = {
    "suturing": TASK1_DIR / "suturing_kinematics_summary.csv",
    "needle_passing": TASK1_DIR / "needle_passing_kinematics_summary.csv",
    "knot_tying": TASK1_DIR / "knot_tying_kinematics_summary.csv",
}

META_FILES = {
    "suturing": TASK2_DIR / "meta_file_Suturing.txt",
    "needle_passing": TASK2_DIR / "meta_file_Needle_Passing.txt",
    "knot_tying": TASK2_DIR / "meta_file_Knot_Tying.txt",
}

OUTPUT_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}


def load_meta_file(meta_path: Path) -> pd.DataFrame:
    """
    Load the JIGSAWS metadata table.

    Expected columns:
    0: filename
    1: skill-level self-proclaimed
    2: skill-level GRS
    3-8: six OSATS component scores
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path, sep=r"\s+|,", engine="python", header=None)

    # Drop any empty columns caused by mixed spacing/tab formatting.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 9:
        raise ValueError(
            f"Expected at least 9 metadata columns, found {df.shape[1]} in {meta_path}"
        )

    df = df.iloc[:, :9].copy()
    df.columns = [
        "filename",
        "experience_level",
        "grs",
        "respect_for_tissue",
        "suture_needle_handling",
        "time_and_motion",
        "flow_of_operation",
        "overall_performance",
        "quality_of_final_product",
    ]

    # Clean filename values for reliable merging.
    df["filename"] = df["filename"].astype(str).str.strip()

    return df


def load_summary_file(summary_path: Path) -> pd.DataFrame:
    """
    Load the motion-feature summary CSV produced by motion_feature_storage.py.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if "file" not in df.columns:
        raise ValueError(f"Expected a 'file' column in {summary_path}")

    # Match the metadata key, e.g. Suturing_B001.txt -> Suturing_B001
    df["filename"] = df["file"].astype(str).str.replace(r"\.txt$", "", regex=True).str.strip()

    return df


def merge_one_dataset(summary_path: Path, meta_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Merge one summary file with its corresponding metadata table.
    """
    summary_df = load_summary_file(summary_path)
    meta_df = load_meta_file(meta_path)

    merged = pd.merge(meta_df, summary_df, on="filename", how="inner")

    # Optional: put the label columns first.
    first_cols = [
        "filename",
        "experience_level",
        "grs",
        "respect_for_tissue",
        "suture_needle_handling",
        "time_and_motion",
        "flow_of_operation",
        "overall_performance",
        "quality_of_final_product",
    ]
    remaining_cols = [c for c in merged.columns if c not in first_cols]
    merged = merged[first_cols + remaining_cols]

    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to {output_path}")

    return merged


def main() -> None:
    for key in SUMMARY_FILES:
        merge_one_dataset(
            summary_path=SUMMARY_FILES[key],
            meta_path=META_FILES[key],
            output_path=OUTPUT_FILES[key],
        )


if __name__ == "__main__":
    main()

