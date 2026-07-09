'''
This script generates scatter plots of OSATS/GRS scores against experience level for each task.
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

EXPERIENCE_MAP = {
    "N": 1,
    "Novice": 1,
    "I": 2,
    "Intermediate": 2,
    "E": 3,
    "Expert": 3,
}

TASK_COLOURS = {
    "suturing": "tab:blue",
    "needle_passing": "tab:orange",
    "knot_tying": "tab:green",
}

TASK_LABELS = {
    "suturing": "Suturing",
    "needle_passing": "Needle passing",
    "knot_tying": "Knot tying",
}

OUTPUT_DIR = TASK2_DIR / "experience_osats_scatter_plots"


def pretty_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


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


def plot_target(df: pd.DataFrame, target: str) -> None:
    """
    Plot one OSATS/GRS target against ordinal experience level for all tasks.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(42)

    plotted_any = False

    for task, task_df in df.groupby("task", sort=False):
        if target not in task_df.columns:
            continue

        pair = task_df[["experience_level_numeric", target]].dropna()
        if pair.empty:
            continue

        # Small jitter so the three ordinal experience groups are visible.
        x = pair["experience_level_numeric"].to_numpy(dtype=float)
        x_jitter = x + rng.normal(0, 0.05, size=len(x))
        y = pair[target].to_numpy(dtype=float)

        ax.scatter(
            x_jitter,
            y,
            s=35,
            alpha=0.75,
            color=TASK_COLOURS[task],
            label=TASK_LABELS[task],
            edgecolors="none",
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Experience level (Novice = 1, Intermediate = 2, Expert = 3)")
    ax.set_ylabel(pretty_name(target))
    ax.set_title(f"{pretty_name(target)} vs Experience Level")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Novice", "Intermediate", "Expert"])
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    out_file = OUTPUT_DIR / f"experience_vs_{target}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_merged_datasets()
    combined = encode_experience(combined)

    for target in TARGET_COLUMNS:
        plot_target(combined, target)


if __name__ == "__main__":
    main()

