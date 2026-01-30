"""
===============================================================================
Targeted Misclassification Extraction from Incorrect Prediction Reports
===============================================================================

Description
-----------
This script performs a **targeted post-analysis of classification errors**
by scanning multiple CSV files that store incorrect predictions
(df_incorrect reports).

Instead of analyzing all misclassifications globally, the algorithm focuses
on **specific true → predicted class pairs of interest**, allowing the user
to investigate recurrent or scientifically relevant confusion patterns
(e.g., morphologically similar species).

For each detected confusion, the script records:
- The image file name;
- The true label;
- The predicted (incorrect) label;
- The cross-validation fold index (k), extracted from the filename.

The consolidated output is returned as a single Pandas DataFrame and can be
saved or further analyzed.

Input and Output
----------------
Input:
- A directory containing multiple CSV files following the pattern:
  Test_*_df_incorrect_kX.csv
- A YAML configuration file specifying:
  - Folder path
  - Target confusion pairs
  - Output CSV path

Output:
- A consolidated CSV file with columns:
  file | true_label | predicted_label | k

===============================================================================
"""

import os
import re
import yaml
import argparse
import pandas as pd


# =============================================================================
# Configuration Handling
# =============================================================================
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
        Configuration dictionary containing:
        - input_folder
        - target_confusions
        - output_csv
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Utility Functions
# =============================================================================
def extract_k_from_filename(filename):
    """
    Extract the cross-validation fold index (k) from a filename.

    Expected filename pattern:
    Test_*_df_incorrect_kX.csv

    Example
    -------
    Test_1_1_DenseNet201_df_incorrect_k3.csv → 3

    Parameters
    ----------
    filename : str
        CSV filename.

    Returns
    -------
    int or None
        Fold index k if found, otherwise None.
    """
    match = re.search(r"_k(\d+)\.csv$", filename)
    return int(match.group(1)) if match else None


# =============================================================================
# Core Algorithm
# =============================================================================
def collect_target_confusions(folder_path, target_pairs):
    """
    Extract specific true → predicted misclassification pairs from multiple
    df_incorrect CSV files.

    The function scans all files in the provided directory that match the
    expected naming pattern and filters rows corresponding to the selected
    confusion pairs.

    Parameters
    ----------
    folder_path : str
        Directory containing df_incorrect CSV files.
    target_pairs : list of tuples
        List of (true_label, predicted_label) pairs to extract.

    Returns
    -------
    pandas.DataFrame
        Consolidated DataFrame with columns:
        - file
        - true_label
        - predicted_label
        - k
    """
    records = []

    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv") and "_df_incorrect_k" in f
    ]

    for csv_file in csv_files:
        k = extract_k_from_filename(csv_file)
        csv_path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(csv_path)

        # Expected columns:
        # file | true_label | predicted_label
        for true_cls, pred_cls in target_pairs:
            subset = df[
                (df["true_label"] == true_cls) &
                (df["predicted_label"] == pred_cls)
            ]

            for _, row in subset.iterrows():
                records.append({
                    "file": row["file"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "k": k
                })

    return pd.DataFrame(records)


# =============================================================================
# Execution Pipeline
# =============================================================================
def run(config):
    """
    Execute the targeted misclassification extraction pipeline.

    Steps
    -----
    1. Load configuration parameters;
    2. Scan input folder for incorrect prediction reports;
    3. Extract selected confusion patterns;
    4. Save consolidated results to CSV.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML.

    Returns
    -------
    None
    """
    input_folder = config["input_folder"]
    target_confusions = [tuple(x) for x in config["target_confusions"]]
    output_csv = config["output_csv"]

    df_errors = collect_target_confusions(
        input_folder,
        target_confusions
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_errors.to_csv(output_csv, index=False)

    print(f"Targeted confusion report saved to: {output_csv}")
    print(f"Total extracted errors: {len(df_errors)}")


# =============================================================================
# Command Line Interface
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract specific true → predicted misclassifications from df_incorrect CSV files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="reports/config_target_confusions.yaml",
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()
    config = load_config(args.config)

    run(config)
