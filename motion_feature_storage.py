from pathlib import Path

import pandas as pd

from motion_feature_extraction import load_kinematic_file, analyse_arm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Suturing kinematics" / "AllGestures"
OUTPUT_CSV = SCRIPT_DIR / "suturing_kinematics_summary.csv"
OUTPUT_CSV = Path("suturing_kinematics_summary.csv")

# Five per-arm metrics to average across slave_left and slave_right.
# If you want a different fifth metric, change this list.
METRICS = [
    ("path_length", "path_length"),
    ("economy_of_motion", "economy_of_motion"),
    ("rms_jerk", "rms_jerk"),
    ("smoothness_score", "smoothness_score"),
    ("duration_s", "duration_s"),
]


def analyse_file_mean_both_arms(file_path: Path) -> dict:
    """
    Load one JIGSAWS kinematic file, compute features for both slave arms,
    and return the mean value across the two arms for each metric.
    """
    data = load_kinematic_file(file_path)

    left = analyse_arm(data, "slave_left")
    right = analyse_arm(data, "slave_right")

    row = {"file": file_path.name}

    for metric_name, _ in METRICS:
        left_key = f"slave_left_{metric_name}"
        right_key = f"slave_right_{metric_name}"

        if left_key not in left or right_key not in right:
            raise KeyError(f"Missing metric keys: {left_key}, {right_key}")

        row[f"mean_{metric_name}"] = (left[left_key] + right[right_key]) / 2.0

    return row


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {DATA_DIR}")

    print("Looking in:", DATA_DIR)
    print("Directory exists:", DATA_DIR.exists())
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {DATA_DIR}")

    rows = []
    for file_path in files:
        try:
            rows.append(analyse_file_mean_both_arms(file_path))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    if not rows:
        raise RuntimeError("No files were processed successfully.")

    df = pd.DataFrame(rows)

    # Optional: order columns consistently
    column_order = ["file"] + [f"mean_{name}" for name, _ in METRICS]
    df = df[column_order]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()