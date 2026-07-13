"""
===========================================================================
ATARI Surgical Skill Assessment GUI
===========================================================================

This Streamlit interface allows a user to:

1. Select a JIGSAWS kinematic trial from a dropdown list.
2. Extract motion features using the Task 1 functions.
3. Predict the six OSATS scores, GRS and experience level.
4. Generate written training feedback.
5. Display the numerical output produced by Create_Feedback.py.
6. Save the standard CSV and text feedback files.

Run from the Task 4 directory using:

    streamlit run GUI.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from Create_Feedback import (
    EXPERIENCE_DISPLAY_NAMES,
    KINEMATIC_FOLDERS,
    OSATS_DISPLAY_NAMES,
    OSATS_TARGETS,
    create_feedback_summary,
    create_model_input,
    extract_motion_features,
    load_training_data,
    predict_scores,
    save_outputs,
    train_final_models,
)


# =========================================================================
# PAGE CONFIGURATION
# =========================================================================

st.set_page_config(
    page_title="ATARI Surgical Skill Assessment",
    page_icon="🩺",
    layout="wide",
)

st.title("ATARI Surgical Skill Assessment")
st.write(
    "Select a JIGSAWS kinematic trial to extract motion features, "
    "predict OSATS scores and generate training feedback."
)


# =========================================================================
# FILE DISCOVERY
# =========================================================================

def discover_kinematic_files() -> dict[str, tuple[Path, str]]:
    """
    Search the three JIGSAWS task folders and return all available .txt files.

    The dictionary key is the user-facing dropdown label. The value contains
    the complete file path and the corresponding task name.
    """
    files: dict[str, tuple[Path, str]] = {}

    for task_name, folder in KINEMATIC_FOLDERS.items():
        if not folder.exists():
            continue

        for file_path in sorted(folder.rglob("*.txt")):
            task_label = task_name.replace("_", " ").title()
            display_label = f"{task_label} — {file_path.stem}"

            files[display_label] = (file_path, task_name)

    return files


# =========================================================================
# MODEL CACHING
# =========================================================================

@st.cache_resource
def load_models():
    """
    Train and cache the final Random Forest models.

    Streamlit only performs this training once per application session,
    rather than retraining every time a different file is selected.
    """
    training_data = load_training_data()

    return train_final_models(training_data)


# =========================================================================
# RESULT FORMATTING
# =========================================================================

def create_results_dataframe(
    input_file: Path,
    task: str,
    motion_features: dict[str, float],
    predicted_osats: dict[str, float],
    predicted_grs: float,
    predicted_experience: str,
    experience_probabilities: dict[str, float],
) -> pd.DataFrame:
    """
    Create the same one-row numerical output format used by
    Create_Feedback.py.
    """
    row: dict[str, object] = {
        "filename": input_file.name,
        "task": task,
        **motion_features,
    }

    for target, score in predicted_osats.items():
        row[f"predicted_{target}"] = score

    row["predicted_grs"] = predicted_grs
    row["predicted_experience_level"] = predicted_experience

    for label in ["N", "I", "E"]:
        row[f"probability_{label}"] = experience_probabilities.get(label, 0.0)

    return pd.DataFrame([row])


def create_display_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw one-row CSV output into a readable two-column table.
    """
    row = results_df.iloc[0]

    display_names = {
        "filename": "Filename",
        "task": "Task",
        "mean_path_length": "Mean path length",
        "mean_economy_of_motion": "Mean movement cost",
        "mean_rms_jerk": "Mean RMS jerk",
        "mean_smoothness_score": "Mean smoothness score",
        "mean_duration_s": "Duration (seconds)",
        "predicted_respect_for_tissue": "Respect for tissue",
        "predicted_suture_needle_handling": "Suture/needle handling",
        "predicted_time_and_motion": "Time and motion",
        "predicted_flow_of_operation": "Flow of operation",
        "predicted_overall_performance": "Overall performance",
        "predicted_quality_of_final_product": "Quality of final product",
        "predicted_grs": "Predicted GRS",
        "predicted_experience_level": "Predicted experience level",
        "probability_N": "Novice probability",
        "probability_I": "Intermediate probability",
        "probability_E": "Expert probability",
    }

    records = []

    for column in results_df.columns:
        value = row[column]

        if column == "predicted_experience_level":
            value = EXPERIENCE_DISPLAY_NAMES.get(str(value), value)

        elif column.startswith("probability_"):
            value = f"{float(value) * 100:.1f}%"

        elif isinstance(value, float):
            value = round(value, 3)

        records.append(
            {
                "Output": display_names.get(
                    column,
                    column.replace("_", " ").title(),
                ),
                "Value": value,
            }
        )

    return pd.DataFrame(records)


# =========================================================================
# USER INTERFACE
# =========================================================================

available_files = discover_kinematic_files()

if not available_files:
    st.error(
        "No JIGSAWS .txt files were found. Check the paths defined in "
        "KINEMATIC_FOLDERS inside Create_Feedback.py."
    )
    st.stop()


selected_label = st.selectbox(
    "Select a JIGSAWS kinematic file",
    options=list(available_files.keys()),
)

selected_file, selected_task = available_files[selected_label]

st.caption(f"Selected file: {selected_file}")


if st.button("Run analysis", type="primary"):
    try:
        with st.spinner("Extracting features and generating predictions..."):
            motion_features = extract_motion_features(selected_file)

            (
                osats_regressor,
                experience_classifier,
                feature_columns,
            ) = load_models()

            model_input = create_model_input(
                motion_features=motion_features,
                task=selected_task,
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
                file_name=selected_file.name,
                task=selected_task,
                motion_features=motion_features,
                predicted_osats=predicted_osats,
                predicted_grs=predicted_grs,
                predicted_experience=predicted_experience,
                experience_probabilities=experience_probabilities,
            )

            csv_path, text_path = save_outputs(
                input_file=selected_file,
                task=selected_task,
                motion_features=motion_features,
                predicted_osats=predicted_osats,
                predicted_grs=predicted_grs,
                predicted_experience=predicted_experience,
                experience_probabilities=experience_probabilities,
                feedback_text=feedback_text,
            )

            results_df = create_results_dataframe(
                input_file=selected_file,
                task=selected_task,
                motion_features=motion_features,
                predicted_osats=predicted_osats,
                predicted_grs=predicted_grs,
                predicted_experience=predicted_experience,
                experience_probabilities=experience_probabilities,
            )

        st.success("Analysis completed.")

        # -------------------------------------------------------------
        # Main summary
        # -------------------------------------------------------------
        st.subheader("Predicted skill summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.metric(
                label="Predicted GRS",
                value=f"{predicted_grs:.2f} / 30",
            )

        with summary_col2:
            st.metric(
                label="Predicted experience level",
                value=EXPERIENCE_DISPLAY_NAMES.get(
                    predicted_experience,
                    predicted_experience,
                ),
            )

        # -------------------------------------------------------------
        # OSATS scores
        # -------------------------------------------------------------
        st.subheader("Predicted OSATS scores")

        osats_columns = st.columns(3)

        for index, target in enumerate(OSATS_TARGETS):
            with osats_columns[index % 3]:
                st.metric(
                    label=OSATS_DISPLAY_NAMES[target],
                    value=f"{predicted_osats[target]:.2f} / 5",
                )

        # -------------------------------------------------------------
        # Experience probabilities
        # -------------------------------------------------------------
        st.subheader("Experience classification probabilities")

        probability_df = pd.DataFrame(
            {
                "Experience level": [
                    "Novice",
                    "Intermediate",
                    "Expert",
                ],
                "Probability": [
                    experience_probabilities.get("N", 0.0),
                    experience_probabilities.get("I", 0.0),
                    experience_probabilities.get("E", 0.0),
                ],
            }
        )

        st.bar_chart(
            probability_df.set_index("Experience level"),
            y="Probability",
        )

        # -------------------------------------------------------------
        # Complete CSV output
        # -------------------------------------------------------------
        st.subheader("Complete analysis output")

        display_df = create_display_table(results_df)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Show raw CSV format"):
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------------
        # Written feedback
        # -------------------------------------------------------------
        st.subheader("Training feedback")
        st.text(feedback_text)

        # -------------------------------------------------------------
        # Download controls
        # -------------------------------------------------------------
        st.subheader("Download results")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="Download results CSV",
                data=results_df.to_csv(index=False),
                file_name=csv_path.name,
                mime="text/csv",
            )

        with download_col2:
            st.download_button(
                label="Download feedback report",
                data=feedback_text,
                file_name=text_path.name,
                mime="text/plain",
            )

        st.caption(f"CSV saved locally to: {csv_path}")
        st.caption(f"Feedback report saved locally to: {text_path}")

    except Exception as error:
        st.error("The analysis could not be completed.")
        st.exception(error)


st.divider()
st.caption(
    "Research and formative training tool. Predictions are based on the "
    "JIGSAWS dataset and are not a standalone basis for clinical credentialing."
)

