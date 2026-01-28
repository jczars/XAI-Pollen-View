"""
Image View Prediction and Dataset Organization Script.

This script performs the following steps:
1. Iterates over k-fold datasets.
2. Builds CSV files listing image paths and class labels.
3. Loads a trained Keras model.
4. Predicts image views (EQUATORIAL or POLAR).
5. Organizes and copies images into view-specific directories.
6. Supports configuration through a YAML file and command-line arguments.

Output Location
---------------
All generated results (organized image folders and CSV files) are stored in:

    /media/jczars/4C22F02A22F01B22/XAI-Pollen-View/BD/CPD1_TEST_A200/

The internal directory structure follows:

    CPD1_TEST_A200/
        k1/
            EQUATORIAL/
                <class_name>/
                    image_001.jpg
            POLAR/
                <class_name>/
                    image_002.jpg
        k2/
        ...
        k10/

This structure allows direct inspection of predicted views and
supports quantitative and qualitative analysis of the results.
"""

import os
import sys
import shutil
import argparse
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras import models

# -------------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------------

# Add the current working directory to PYTHONPATH
# This allows local modules to be imported correctly
sys.path.insert(0, os.getcwd())

# Import custom project modules
from models import get_data, utils, sound_test_finalizado


# -------------------------------------------------------------------------
# Dataset initialization
# -------------------------------------------------------------------------

def initial(params, k):
    """
    Initialize source and destination directories for a given fold.

    Parameters
    ----------
    params : dict
        Dictionary containing configuration parameters.
        Expected keys:
        - 'bd_src': base source directory
        - 'bd_dst': base destination directory
    k : int
        Fold index (k-fold cross-validation).

    Returns
    -------
    tuple of str
        bd_src : Path to the source directory for fold k.
        bd_dst : Path to the destination directory for fold k.
    """
    bd_src = f"{params['bd_src']}/k{k}"
    print(f"[INFO] Source directory: {bd_src}")

    bd_dst = f"{params['bd_dst']}/k{k}"
    print(f"[INFO] Destination directory: {bd_dst}")

    os.makedirs(bd_dst, exist_ok=True)

    return bd_src, bd_dst


# -------------------------------------------------------------------------
# Dataset creation and loading
# -------------------------------------------------------------------------

def create_dataSet(bd_src, bd_dst):
    """
    Create a CSV file listing all images and their class labels.

    The expected directory structure is:
        bd_src/
            class_1/
                image_1.jpg
            class_2/
                image_2.jpg

    Parameters
    ----------
    bd_src : str
        Path to the dataset source directory.
    bd_dst : str
        Path to the destination directory where the CSV will be saved.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    data = pd.DataFrame(columns=['file', 'labels'])
    counter = 0

    categories = os.listdir(bd_src)

    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(bd_src, category)

        if not os.path.isdir(category_path):
            print(f"[WARNING] {category_path} is not a directory. Skipping.")
            continue

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)

            if os.path.isfile(file_path):
                data.loc[counter] = [file_path, category]
                counter += 1

    csv_path = f"{bd_dst}data.csv"
    data.to_csv(csv_path, index=False, header=True)

    print(f"[INFO] Dataset CSV saved at: {csv_path}")

    return csv_path


def read_data_csv(csv_path):
    """
    Load a dataset CSV and generate a summary CSV by class label.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset containing file paths and labels.
    """
    data_csv = pd.read_csv(csv_path)

    summary_csv = csv_path.replace('.csv', '_summary.csv')
    label_counts = data_csv.groupby('labels').size().reset_index(name='count')
    label_counts.to_csv(summary_csv, index=False, header=True)

    print("[INFO] Dataset summary:")
    print(data_csv.groupby('labels').count())

    return data_csv


# -------------------------------------------------------------------------
# Prediction and post-processing
# -------------------------------------------------------------------------

def predict_data_generator(test_data_generator, model, categories, batch_size, verbose=2):
    """
    Predict image views using a trained model and summarize predictions.

    Parameters
    ----------
    test_data_generator : keras.preprocessing iterator
        Generator providing test images.
    model : keras.Model
        Trained Keras model.
    categories : list
        Output category names corresponding to model predictions.
    batch_size : int
        Batch size used during prediction.
    verbose : int, optional
        Verbosity mode for model prediction.

    Returns
    -------
    tuple
        df : pandas.DataFrame
            DataFrame containing image paths, predicted labels, views, and classes.
        quantidade_por_vista_classe : pandas.DataFrame
            Aggregated count of images per (vista, classe).
    """
    filenames = test_data_generator.filenames
    df = pd.DataFrame(filenames, columns=['file'])

    print(f"[INFO] Predicting {len(filenames)} images")
    print(f"[INFO] Batch size: {batch_size}")

    y_preds = model.predict(test_data_generator, verbose=verbose)
    y_pred = np.argmax(y_preds, axis=1)
    df['y_pred'] = y_pred

    vistas = []
    classes = []

    for _, row in df.iterrows():
        vista_pred = categories[row['y_pred']]
        vista = vista_pred.split('_')[0]
        classe = row['file'].split('/')[-2]

        vistas.append(vista)
        classes.append(classe)

    df['vista'] = vistas
    df['classe'] = classes

    quantidade_por_vista_classe = (
        df.groupby(['vista', 'classe'])
        .size()
        .reset_index(name='quantidade')
    )

    return df, quantidade_por_vista_classe


def copy_images_by_vista(bd_dst, df):
    """
    Copy images into directories organized by predicted view.

    Parameters
    ----------
    bd_dst : str
        Base destination directory.
    df : pandas.DataFrame
        DataFrame containing image paths and predicted views.
    """
    equatorial_dir = f"{bd_dst}/EQUATORIAL"
    polar_dir = f"{bd_dst}/POLAR"

    os.makedirs(equatorial_dir, exist_ok=True)
    os.makedirs(polar_dir, exist_ok=True)

    for _, row in df.iterrows():
        file_path = row['file']
        vista = row['vista'].lower()
        class_name = file_path.split('/')[-2]

        if vista == 'equatorial':
            destination_folder = os.path.join(equatorial_dir, class_name)
        elif vista == 'polar':
            destination_folder = os.path.join(polar_dir, class_name)
        else:
            continue

        os.makedirs(destination_folder, exist_ok=True)

        destination_path = os.path.join(destination_folder, os.path.basename(file_path))

        try:
            shutil.copy(file_path, destination_path)
        except Exception as e:
            print(f"[ERROR] Could not copy {file_path}: {e}")


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def run(params):
    """
    Execute the full prediction and dataset organization pipeline.

    Parameters
    ----------
    params : dict
        Configuration dictionary loaded from a YAML file.
    """
    print(f"[INFO] Parameters: {params}")

    image_size = params['image_size']
    input_shape = (image_size, image_size)

    categories_vistas = sorted(os.listdir(params['path_labels']))

    print("[INFO] Loading trained model...")
    model = models.load_model(params['path_model'])
    model.summary()

    for k in range(1, 11):
        print(f"\n[INFO] Processing fold k={k}")

        bd_src, bd_dst = initial(params, k)
        csv_data = create_dataSet(bd_src, bd_dst)
        data = read_data_csv(csv_data)

        test_data_generator = get_data.load_data_test(data, input_shape)

        df_vistas, df_quantidade = predict_data_generator(
            test_data_generator,
            model,
            categories_vistas,
            params['batch_size']
        )

        copy_images_by_vista(bd_dst, df_vistas)

    print("[INFO] Process finished successfully.")


# -------------------------------------------------------------------------
# Argument parsing and configuration
# -------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict image views and organize datasets by EQUATORIAL and POLAR."
    )

    parser.add_argument(
        '--config',
        type=str,
        help="Path to the YAML configuration file."
    )

    return parser.parse_args()


def load_config(config_path="config.yaml"):
    """
    Load configuration parameters from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    config_file = args.config if args.config else 'phase3/config_separeted.yaml'
    params = load_config(config_file)

    debug = True

    if debug:
        run(params)
        sound_test_finalizado.beep(2)
    else:
        try:
            run(params)
            sound_test_finalizado.beep(2, "[INFO] Successfully finished!")
        except Exception as e:
            sound_test_finalizado.beep(2, f"[ERROR] {str(e)}")
