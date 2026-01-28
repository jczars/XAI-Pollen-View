"""
===============================================================================
Confusion Matrix Post-Analysis and Error Interpretation
===============================================================================

Description
-----------
This script performs a **post-hoc analytical inspection of a confusion matrix**
generated during the evaluation of a classification model.

Given a confusion matrix stored as a CSV file, the algorithm extracts
high-level error patterns to support qualitative and quantitative discussion
in experimental analysis. Specifically, it identifies:

1. Classes with the highest number of misclassifications;
2. The most frequent confusion pairs (true class → predicted class);
3. Classes that most frequently receive incorrect predictions.

The results are consolidated into a structured **TXT report**, intended
for direct inclusion in the discussion or error analysis section of a
scientific article.

Input and Output
----------------
Input:
- A confusion matrix stored as a CSV file.

Output:
- A human-readable TXT report summarizing classification errors.

Configuration is handled via an external YAML file.

===============================================================================
"""

import argparse
import os
import sys
import yaml
import pandas as pd

# Add project root to PYTHONPATH
sys.path.insert(0, os.getcwd())

from models import sound_test_finalizado


def load_config(config_path):
    """
    Load configuration parameters from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing configuration parameters such as:
        - confusion_matrix_csv
        - output_txt_path
        - top_k_confusions (optional)
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_confusion_matrix(csv_path, top_k=5):
    """
    Perform analytical inspection of a confusion matrix.

    This function extracts three complementary perspectives of classification
    errors:
    1. Total number of errors per true class;
    2. Most frequent confusion pairs (True → Predicted);
    3. Aggregated number of incorrect predictions received by each class.

    Parameters
    ----------
    csv_path : str
        Path to the confusion matrix CSV file.
    top_k : int, optional
        Number of top confusion pairs to retain (default is 5).

    Returns
    -------
    errors_per_class : pandas.Series
        Number of misclassifications per true class.
    top_confusions : pandas.DataFrame
        Top-k most frequent confusion pairs.
    wrong_predictions : pandas.Series
        Number of incorrect predictions aggregated by predicted class.
    """
    # Load confusion matrix
    df = pd.read_csv(csv_path, index_col=0)
    df = df.apply(pd.to_numeric)

    # ------------------------------------------------------------------
    # 1) Errors per true class (row sum minus diagonal)
    # ------------------------------------------------------------------
    errors_per_class = df.sum(axis=1) - df.values.diagonal()
    errors_per_class = errors_per_class.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # 2) Confusion pairs (True Class → Predicted Class)
    # ------------------------------------------------------------------
    confusions = []
    for true_class in df.index:
        for pred_class in df.columns:
            if true_class != pred_class and df.loc[true_class, pred_class] > 0:
                confusions.append(
                    (true_class, pred_class, int(df.loc[true_class, pred_class]))
                )

    confusions_df = pd.DataFrame(
        confusions, columns=["True Class", "Predicted Class", "Count"]
    )

    top_confusions = (
        confusions_df
        .sort_values("Count", ascending=False)
        .head(top_k)
    )

    # ------------------------------------------------------------------
    # 3) Wrong predictions aggregated by predicted class
    # ------------------------------------------------------------------
    wrong_predictions = (
        confusions_df
        .groupby("Predicted Class")["Count"]
        .sum()
        .sort_values(ascending=False)
    )

    return errors_per_class, top_confusions, wrong_predictions


def save_report_txt(
    output_path,
    errors_per_class,
    top_confusions,
    wrong_predictions,
    csv_path
):
    """
    Save the confusion matrix analysis as a structured TXT report.

    Parameters
    ----------
    output_path : str
        Path where the TXT report will be saved.
    errors_per_class : pandas.Series
        Misclassification count per true class.
    top_confusions : pandas.DataFrame
        Top confusion pairs.
    wrong_predictions : pandas.Series
        Aggregated incorrect predictions per predicted class.
    csv_path : str
        Path to the original confusion matrix CSV file.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("CONFUSION MATRIX ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Source CSV: {csv_path}\n\n")

        f.write("1) Classes with the highest number of classification errors:\n")
        f.write("-" * 40 + "\n")
        for cls, val in errors_per_class.items():
            f.write(f"{cls}: {val}\n")

        f.write("\n2) Most frequent confusion pairs (True → Predicted):\n")
        f.write("-" * 40 + "\n")
        for _, row in top_confusions.iterrows():
            f.write(
                f"{row['True Class']} → {row['Predicted Class']}: {row['Count']}\n"
            )

        f.write("\n3) Classes receiving the highest number of wrong predictions:\n")
        f.write("-" * 40 + "\n")
        for cls, val in wrong_predictions.items():
            f.write(f"{cls}: {val}\n")

        f.write("\nEnd of report.\n")


def run(config):
    """
    Main execution function.

    This function orchestrates the analysis pipeline by:
    - Loading the confusion matrix;
    - Performing error analysis;
    - Saving a consolidated textual report.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    None
    """
    csv_path = config["confusion_matrix_csv"]
    output_txt = config["output_txt_path"]
    top_k = config.get("top_k_confusions", 5)

    errors_per_class, top_confusions, wrong_predictions = analyze_confusion_matrix(
        csv_path, top_k
    )

    save_report_txt(
        output_txt,
        errors_per_class,
        top_confusions,
        wrong_predictions,
        csv_path
    )

    print(f"Report saved to: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a confusion matrix and generate a textual error report."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="reports/config_matrix_analyze.yaml",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    run(config)

    # Audible notification indicating successful execution
    sound_test_finalizado.beep(2)
