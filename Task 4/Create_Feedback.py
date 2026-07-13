"""
===========================================================================
Create Surgical Training Feedback
===========================================================================

Purpose
-------
This script analyses one JIGSAWS kinematic trial and produces:

Uses Task 1 functions to:
    - extracted motion features 
Uses Random Forest models trained on the merged Task 2 datasets to predict:
    - six predicted OSATS component scores 
    - predicted Global Rating Score (GRS)
    - predicted experience level
New code to output task 4 specific feedback report:
    - written training feedback highlighting strengths and areas to improve
    - i set three thresholds for each OSATS component score to generate 
      feedback and wrote recommendations for each score range

The script uses the same task-aware Random Forest architecture as
Classifier_2.py. It trains final models using all labelled trials from the
three merged Task 2 datasets, then applies those models to a selected raw
kinematic file.

This implementation is intended for research and training demonstrations.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================================================================
# PROJECT PATHS
# =========================================================================

TASK4_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TASK4_DIR.parent

TASK1_DIR = PROJECT_DIR / "Task 1"
TASK2_DIR = PROJECT_DIR / "Task 2"

# Allow functions in Task 1 to be imported.
if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))

from motion_feature_extraction import analyse_arm, load_kinematic_file


# =========================================================================
# INPUT CONFIGURATION
# =========================================================================

# Used when the script is run without the --file command-line argument.
DEFAULT_INPUT_FILE = "Suturing_H004.txt"

KINEMATIC_FOLDERS = {
    "suturing": TASK1_DIR / "Suturing kinematics" / "AllGestures",
    "needle_passing": TASK1_DIR / "Needle_passing kinematics" / "AllGestures",
    "knot_tying": TASK1_DIR / "Knot_Tying kinematics" / "AllGestures",
}

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

OUTPUT_DIR = TASK4_DIR / "feedback_outputs"


# These names must match the feature columns created in Task 1.
MOTION_METRICS = [
    "path_length",
    "economy_of_motion",
    "rms_jerk",
    "smoothness_score",
    "duration_s",
]

OSATS_TARGETS = [
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]

OSATS_DISPLAY_NAMES = {
    "respect_for_tissue": "Respect for tissue",
    "suture_needle_handling": "Suture/needle handling",
    "time_and_motion": "Time and motion",
    "flow_of_operation": "Flow of operation",
    "overall_performance": "Overall performance",
    "quality_of_final_product": "Quality of final product",
}

EXPERIENCE_DISPLAY_NAMES = {
    "N": "Novice",
    "I": "Intermediate",
    "E": "Expert",
}


# =========================================================================
# FILE AND TASK IDENTIFICATION
# =========================================================================

def normalise_file_name(file_name: str) -> str:
    """
    Ensure that the requested kinematic filename has a .txt extension.
    """
    path = Path(file_name)

    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

    return path.name


def infer_task_from_filename(file_name: str) -> str:
    """
    Infer the surgical task from the kinematic filename.
    """
    name = Path(file_name).stem.lower()

    if name.startswith("suturing"):
        return "suturing"

    if name.startswith("needle"):
        return "needle_passing"

    if name.startswith("knot"):
        return "knot_tying"

    raise ValueError(
        f"Could not infer the task from '{file_name}'. "
        "Expected a filename beginning with Suturing, Needle or Knot."
    )


def find_kinematic_file(file_name_or_path: str) -> tuple[Path, str]:
    """
    Find a requested JIGSAWS kinematic file.

    A complete path may be supplied. Otherwise, the script searches the
    three Task 1 kinematic folders.
    """
    supplied_path = Path(file_name_or_path)

    if supplied_path.exists():
        task = infer_task_from_filename(supplied_path.name)
        return supplied_path.resolve(), task

    file_name = normalise_file_name(file_name_or_path)
    task = infer_task_from_filename(file_name)

    expected_path = KINEMATIC_FOLDERS[task] / file_name

    if expected_path.exists():
        return expected_path.resolve(), task

    # Case-insensitive fallback search.
    target_stem = Path(file_name).stem.lower()

    for folder in KINEMATIC_FOLDERS.values():
        if not folder.exists():
            continue

        for candidate in folder.rglob("*.txt"):
            if candidate.stem.lower() == target_stem:
                detected_task = infer_task_from_filename(candidate.name)
                return candidate.resolve(), detected_task

    raise FileNotFoundError(
        f"Could not find '{file_name_or_path}' in the JIGSAWS folders."
    )


# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def extract_motion_features(file_path: Path) -> dict[str, float]:
    """
    Extract the motion metrics for both slave instruments and calculate the
    mean value across the left and right slave arms.

    The output column names match the Task 1 summary CSV files.
    """
    data = load_kinematic_file(file_path)

    left_results = analyse_arm(data, "slave_left")
    right_results = analyse_arm(data, "slave_right")

    features: dict[str, float] = {}

    for metric in MOTION_METRICS:
        left_key = f"slave_left_{metric}"
        right_key = f"slave_right_{metric}"

        if left_key not in left_results:
            raise KeyError(
                f"analyse_arm() did not return the expected key: {left_key}"
            )

        if right_key not in right_results:
            raise KeyError(
                f"analyse_arm() did not return the expected key: {right_key}"
            )

        features[f"mean_{metric}"] = float(
            (left_results[left_key] + right_results[right_key]) / 2.0
        )

    return features


# =========================================================================
# TRAINING DATA
# =========================================================================

def load_training_data() -> pd.DataFrame:
    """
    Load and combine the merged datasets from Task 2.
    """
    frames = []

    for task_name, file_path in MERGED_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Merged training file not found: {file_path}")

        df = pd.read_csv(file_path)
        df["task"] = task_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    for target in OSATS_TARGETS:
        if target not in combined.columns:
            raise ValueError(f"Missing OSATS target column: {target}")

        combined[target] = pd.to_numeric(
            combined[target],
            errors="coerce",
        )

    if "experience_level" not in combined.columns:
        raise ValueError(
            "The merged training data do not contain 'experience_level'."
        )

    combined["experience_level"] = (
        combined["experience_level"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    combined = combined[
        combined["experience_level"].isin(["N", "I", "E"])
    ].copy()

    # Remove rows with missing target values.
    combined = combined.dropna(
        subset=OSATS_TARGETS + ["experience_level"]
    ).reset_index(drop=True)

    return combined


def add_task_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the task identity so that the Random Forest can learn
    task-specific relationships while still using all three datasets.
    """
    output = df.copy()

    for task_name in KINEMATIC_FOLDERS:
        output[f"task_{task_name}"] = (
            output["task"] == task_name
        ).astype(int)

    return output


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select only motion features and one-hot task identity columns.

    Metadata, expert scores and labels are explicitly excluded.
    """
    excluded_columns = {
        "file",
        "filename",
        "task",
        "experience_level",
        "grs",
        *OSATS_TARGETS,
    }

    feature_columns = []

    for column in df.columns:
        if column in excluded_columns:
            continue

        if column.startswith("task_"):
            feature_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No usable model input features were found.")

    return feature_columns


# =========================================================================
# RANDOM FOREST MODELS
# =========================================================================

def train_final_models(
    training_data: pd.DataFrame,
) -> tuple[
    RandomForestRegressor,
    RandomForestClassifier,
    list[str],
]:
    """
    Train final Random Forest models using all available labelled trials.

    One multi-output regressor predicts all six OSATS component scores.
    A separate classifier predicts experience level.
    """
    prepared_data = add_task_features(training_data)
    feature_columns = get_feature_columns(prepared_data)

    X = prepared_data[feature_columns]
    y_osats = prepared_data[OSATS_TARGETS]
    y_experience = prepared_data["experience_level"]

    osats_regressor = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    experience_classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    osats_regressor.fit(X, y_osats)
    experience_classifier.fit(X, y_experience)

    return osats_regressor, experience_classifier, feature_columns


def create_model_input(
    motion_features: dict[str, float],
    task: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Build one model-input row with the same column structure used in training.
    """
    row: dict[str, float] = dict(motion_features)

    for task_name in KINEMATIC_FOLDERS:
        row[f"task_{task_name}"] = int(task == task_name)

    input_df = pd.DataFrame([row])

    # Add any missing model columns as zero and preserve the training order.
    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = 0.0

    return input_df[feature_columns]


# =========================================================================
# PREDICTION
# =========================================================================

def predict_scores(
    model_input: pd.DataFrame,
    osats_regressor: RandomForestRegressor,
    experience_classifier: RandomForestClassifier,
) -> tuple[dict[str, float], float, str, dict[str, float]]:
    """
    Predict the six OSATS scores, GRS and experience level.

    Continuous OSATS predictions are clipped to the valid range of 1 to 5.
    """
    raw_osats_prediction = osats_regressor.predict(model_input)[0]

    clipped_osats_prediction = np.clip(
        raw_osats_prediction,
        1.0,
        5.0,
    )

    predicted_osats = {
        target: float(score)
        for target, score in zip(
            OSATS_TARGETS,
            clipped_osats_prediction,
        )
    }

    predicted_grs = float(sum(predicted_osats.values()))
    predicted_experience = str(
        experience_classifier.predict(model_input)[0]
    )

    class_probabilities = experience_classifier.predict_proba(model_input)[0]

    experience_probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            experience_classifier.classes_,
            class_probabilities,
        )
    }

    return (
        predicted_osats,
        predicted_grs,
        predicted_experience,
        experience_probabilities,
    )


# =========================================================================
# FEEDBACK GENERATION
# =========================================================================

def feedback_for_score(target: str, score: float) -> str:
    """
    Convert an OSATS prediction into a concise training recommendation.
    """
    feedback_rules = {
        "respect_for_tissue": {
            "low": (
                "Frequently used unnecessary force on tissue."
            ),
            "medium": (
                "Careful tissue handling but occasionally caused inadvertent damage."
            ),
            "high": (
                "Consistent appropriate tissue handling."
            ),
        },
        "suture_needle_handling": {
            "low": (
                "Awkward and unsure with repeated entanglement and poor knot tying."
            ),
            "medium": (
                "Majority of knots placed correctly with appropriate tension."
            ),
            "high": (
                "Excellent suture control"
            ),
        },
        "time_and_motion": {
            "low": (
                "Reduce unnecessary path length, repeated corrections and "
                "inactive instrument movement."
            ),
            "medium": (
                "Look for opportunities to shorten instrument trajectories "
                "and complete movements more directly."
            ),
            "high": (
                "The trial demonstrates efficient use of time and motion."
            ),
        },
        "flow_of_operation": {
            "low": (
                "Practise the procedure as a sequence of planned steps to "
                "reduce pauses, reversals and disrupted transitions."
            ),
            "medium": (
                "Improve transitions between procedural steps and reduce "
                "hesitation before the next action."
            ),
            "high": (
                "The procedural sequence appears fluent and well organised."
            ),
        },
        "overall_performance": {
            "low": (
                "Focus on reliable execution of the full task before attempting "
                "to increase speed."
            ),
            "medium": (
                "Performance is developing; prioritise consistency across the "
                "complete procedure."
            ),
            "high": (
                "Overall technical execution appears strong."
            ),
        },
        "quality_of_final_product": {
            "low": (
                "Review the final task result and practise the steps most "
                "directly affecting accuracy and completion quality."
            ),
            "medium": (
                "Improve consistency in the final result while maintaining "
                "controlled instrument motion."
            ),
            "high": (
                "The predicted final-product quality is strong."
            ),
        },
    }

    if score < 2.5:
        level = "low"
    elif score < 4.0:
        level = "medium"
    else:
        level = "high"

    return feedback_rules[target][level]


def create_feedback_summary(
    file_name: str,
    task: str,
    motion_features: dict[str, float],
    predicted_osats: dict[str, float],
    predicted_grs: float,
    predicted_experience: str,
    experience_probabilities: dict[str, float],
) -> str:
    """
    Generate a readable training-feedback report from the predictions.
    """
    sorted_scores = sorted(
        predicted_osats.items(),
        key=lambda item: item[1],
    )

    lowest_scores = sorted_scores[:2]
    highest_scores = sorted_scores[-2:][::-1]

    lines = [
        "SURGICAL SKILL FEEDBACK REPORT",
        "=" * 40,
        "",
        f"Trial: {file_name}",
        f"Task: {task.replace('_', ' ').title()}",
        "",
        "PREDICTED SUMMARY",
        "-" * 40,
        f"Predicted GRS: {predicted_grs:.2f} / 30",
        (
            "Predicted experience level: "
            f"{EXPERIENCE_DISPLAY_NAMES.get(predicted_experience, predicted_experience)}"
        ),
        "",
        "Experience classification probabilities:",
    ]

    for label in ["N", "I", "E"]:
        probability = experience_probabilities.get(label, 0.0)

        lines.append(
            f"  {EXPERIENCE_DISPLAY_NAMES[label]}: "
            f"{probability * 100:.1f}%"
        )

    lines.extend(
        [
            "",
            "PREDICTED OSATS SCORES",
            "-" * 40,
        ]
    )

    for target in OSATS_TARGETS:
        lines.append(
            f"{OSATS_DISPLAY_NAMES[target]}: "
            f"{predicted_osats[target]:.2f} / 5"
        )

    lines.extend(
        [
            "",
            "MAIN STRENGTHS",
            "-" * 40,
        ]
    )

    for target, score in highest_scores:
        lines.append(
            f"- {OSATS_DISPLAY_NAMES[target]} ({score:.2f}/5): "
            f"{feedback_for_score(target, score)}"
        )

    lines.extend(
        [
            "",
            "PRIORITY AREAS FOR IMPROVEMENT",
            "-" * 40,
        ]
    )

    for target, score in lowest_scores:
        lines.append(
            f"- {OSATS_DISPLAY_NAMES[target]} ({score:.2f}/5): "
            f"{feedback_for_score(target, score)}"
        )

    lines.extend(
        [
            "",
            "MOTION FEATURES",
            "-" * 40,
        ]
    )

    for feature_name, value in motion_features.items():
        display_name = feature_name.replace("_", " ").title()
        lines.append(f"{display_name}: {value:.6g}")

    lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "-" * 40,
            (
                "These outputs are model estimates derived from the JIGSAWS "
                "training dataset. They are suitable for research and formative "
                "training feedback, but not as a standalone basis for clinical "
                "credentialing."
            ),
        ]
    )

    return "\n".join(lines)


# =========================================================================
# OUTPUT STORAGE
# =========================================================================

def save_outputs(
    input_file: Path,
    task: str,
    motion_features: dict[str, float],
    predicted_osats: dict[str, float],
    predicted_grs: float,
    predicted_experience: str,
    experience_probabilities: dict[str, float],
    feedback_text: str,
) -> tuple[Path, Path]:
    """
    Save the numerical results to CSV and the feedback report to a text file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_stem = input_file.stem

    csv_path = OUTPUT_DIR / f"{output_stem}_feedback_results.csv"
    text_path = OUTPUT_DIR / f"{output_stem}_feedback_report.txt"

    output_row: dict[str, object] = {
        "filename": input_file.name,
        "task": task,
        **motion_features,
    }

    for target, score in predicted_osats.items():
        output_row[f"predicted_{target}"] = score

    output_row["predicted_grs"] = predicted_grs
    output_row["predicted_experience_level"] = predicted_experience

    for label in ["N", "I", "E"]:
        output_row[f"probability_{label}"] = (
            experience_probabilities.get(label, 0.0)
        )

    pd.DataFrame([output_row]).to_csv(csv_path, index=False)
    text_path.write_text(feedback_text, encoding="utf-8")

    return csv_path, text_path


# =========================================================================
# MAIN PROGRAM
# =========================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract motion features and generate predicted surgical-skill "
            "feedback for one JIGSAWS kinematic file."
        )
    )

    parser.add_argument(
        "--file",
        default=DEFAULT_INPUT_FILE,
        help=(
            "Kinematic filename or complete file path. "
            f"Default: {DEFAULT_INPUT_FILE}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_file, task = find_kinematic_file(args.file)

    print(f"Analysing: {input_file.name}")
    print(f"Task: {task}")

    motion_features = extract_motion_features(input_file)

    training_data = load_training_data()

    (
        osats_regressor,
        experience_classifier,
        feature_columns,
    ) = train_final_models(training_data)

    model_input = create_model_input(
        motion_features=motion_features,
        task=task,
        feature_columns=feature_columns,
    )

    (
        predicted_osats,
        predicted_grs,
        predicted_experience,
        experience_probabilities,
    ) = predict_scores(
        model_input=model_input,
        osats_regressor=osats_regressor,
        experience_classifier=experience_classifier,
    )

    feedback_text = create_feedback_summary(
        file_name=input_file.name,
        task=task,
        motion_features=motion_features,
        predicted_osats=predicted_osats,
        predicted_grs=predicted_grs,
        predicted_experience=predicted_experience,
        experience_probabilities=experience_probabilities,
    )

    csv_path, text_path = save_outputs(
        input_file=input_file,
        task=task,
        motion_features=motion_features,
        predicted_osats=predicted_osats,
        predicted_grs=predicted_grs,
        predicted_experience=predicted_experience,
        experience_probabilities=experience_probabilities,
        feedback_text=feedback_text,
    )

    print()
    print(feedback_text)
    print()
    print(f"Saved numerical results to: {csv_path}")
    print(f"Saved feedback report to: {text_path}")


if __name__ == "__main__":
    main()

