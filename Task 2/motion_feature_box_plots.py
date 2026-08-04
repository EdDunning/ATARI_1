from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
TASK2_DIR = HERE.parent / "Task 2"

TASK_NAME = "knot_tying"

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

OUTPUT_DIR = HERE / "box_plots" / TASK_NAME

EXPERIENCE_ORDER = ["N", "I", "E"]

EXPERIENCE_LABELS = {
    "N": "Novice",
    "I": "Intermediate",
    "E": "Expert",
}

TARGET_COLUMNS = {
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
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_task_data(task_name: str) -> pd.DataFrame:
    if task_name not in MERGED_FILES:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Choose from: {list(MERGED_FILES.keys())}"
        )

    file_path = MERGED_FILES[task_name]

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    df = pd.read_csv(file_path)

    if "experience_level" not in df.columns:
        raise ValueError(
            "The dataset must contain an 'experience_level' column."
        )

    df["experience_level"] = (
        df["experience_level"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["experience_level"].isin(EXPERIENCE_ORDER)
    ].copy()

    return df


def get_motion_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = []

    for column in df.columns:
        if column in TARGET_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No numeric motion feature columns were found.")

    return feature_columns


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def create_box_plot(
    df: pd.DataFrame,
    feature: str,
    task_name: str,
) -> Path:
    plot_data = []

    for experience in EXPERIENCE_ORDER:
        values = (
            df.loc[df["experience_level"] == experience, feature]
            .dropna()
            .to_numpy()
        )

        plot_data.append(values)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(
        plot_data,
        tick_labels=[
            EXPERIENCE_LABELS[level]
            for level in EXPERIENCE_ORDER
        ],
        showmeans=True,
    )

    ax.set_title(
        f"{feature.replace('_', ' ').title()} by Experience Level\n"
        f"{task_name.replace('_', ' ').title()}"
    )

    ax.set_xlabel("Experience level")
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{feature}_box_plot.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_task_data(TASK_NAME)
    motion_features = get_motion_feature_columns(df)

    print(f"Task: {TASK_NAME}")
    print(f"Trials: {len(df)}")
    print(f"Motion features: {motion_features}")

    for feature in motion_features:
        output_path = create_box_plot(
            df=df,
            feature=feature,
            task_name=TASK_NAME,
        )

        print(f"Saved: {output_path}")

    print(f"\nAll box plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()