'''
This script generates scatter plots for the strongest 6 feature-target pairs based on Spearman correlation
'''
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Suturing kinematics" / "AllGestures"
OUTPUT_CSV = SCRIPT_DIR / "suturing_kinematics_summary.csv"
OUTPUT_CSV = Path("suturing_kinematics_summary.csv")

MERGED_FILES = {
    "suturing": SCRIPT_DIR / "suturing_merged.csv",
    "needle_passing": SCRIPT_DIR / "needle_passing_merged.csv",
    "knot_tying": SCRIPT_DIR / "knot_tying_merged.csv",
}

CORRELATION_FILE = SCRIPT_DIR / "motion_feature_correlations_by_task.csv"
OUTPUT_DIR = SCRIPT_DIR / "motion_feature_osats_scatter_plots"

# Number of strongest feature-target pairs to plot.
TOP_N_PLOTS = 6

# Use fixed colours so each dataset is always identifiable.
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


def pretty_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def load_merged_data() -> dict[str, pd.DataFrame]:
    data = {}

    for task, file_path in MERGED_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing merged file: {file_path}")

        df = pd.read_csv(file_path)
        df["task"] = task
        data[task] = df

    return data


def load_correlation_table() -> pd.DataFrame:
    if not CORRELATION_FILE.exists():
        raise FileNotFoundError(f"Missing correlation file: {CORRELATION_FILE}")

    df = pd.read_csv(CORRELATION_FILE)

    required_cols = {"group", "feature", "target", "spearman_rho", "abs_spearman_rho"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Correlation file is missing columns: {sorted(missing)}")

    return df


def select_top_pairs(corr_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select the strongest unique feature-target pairs based on absolute
    Spearman correlation. If a pair appears in multiple task groups, the
    row with the largest absolute Spearman value is retained.
    """
    ranked = (
        corr_df.sort_values("abs_spearman_rho", ascending=False)
        .drop_duplicates(subset=["feature", "target"], keep="first")
        .head(n)
        .copy()
    )

    return ranked.reset_index(drop=True)


def plot_pair(pair_row: pd.Series, data_by_task: dict[str, pd.DataFrame]) -> None:
    feature = pair_row["feature"]
    target = pair_row["target"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_any = False

    for task, df in data_by_task.items():
        if feature not in df.columns or target not in df.columns:
            continue

        pair_df = df[[feature, target]].dropna()
        if pair_df.empty:
            continue

        ax.scatter(
            pair_df[feature],
            pair_df[target],
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

    task_name = pretty_name(pair_row["group"])
    rho = pair_row["spearman_rho"]
    pearson_r = pair_row["pearson_r"]
    n = int(pair_row["n"])

    ax.set_xlabel(pretty_name(feature))
    ax.set_ylabel(pretty_name(target))
    ax.set_title(
        f"{pretty_name(feature)} vs {pretty_name(target)}\n"
        f"Top task: {task_name} | Spearman ρ = {rho:.3f} | Pearson r = {pearson_r:.3f} | n = {n}"
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    safe_feature = feature.replace("/", "_")
    safe_target = target.replace("/", "_")
    out_file = OUTPUT_DIR / f"{safe_feature}_vs_{safe_target}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def main() -> None:
    data_by_task = load_merged_data()
    corr_df = load_correlation_table()

    top_pairs = select_top_pairs(corr_df, TOP_N_PLOTS)

    # Save the selected pairs for reference.
    selected_pairs_file = OUTPUT_DIR / "selected_pairs.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_pairs.to_csv(selected_pairs_file, index=False)

    print("Selected top pairs:")
    print(top_pairs[["group", "feature", "target", "spearman_rho", "abs_spearman_rho"]].to_string(index=False))
    print()

    for _, row in top_pairs.iterrows():
        plot_pair(row, data_by_task)


if __name__ == "__main__":
    main()

