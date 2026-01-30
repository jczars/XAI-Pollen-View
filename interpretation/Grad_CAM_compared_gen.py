import argparse
import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# PROJECT PATH SETUP
# =========================================================
# Ensure that the project root directory is included in
# PYTHONPATH so that internal modules can be imported
# regardless of where the script is executed from.
sys.path.insert(0, os.getcwd())
print(sys.path)

# =========================================================
# PROJECT-SPECIFIC IMPORTS
# =========================================================
# grad_cam_lib contains all utilities related to:
# - model loading
# - prediction
# - Grad-CAM, Grad-CAM++ and Score-CAM generation
from models import grad_cam_lib as cam
from models.grad_cam_lib import (
    predict_run,
    make_gradcam_heatmap,
    grad_cam_plus,
    ScoreCam,
    superimpose_heatmap_on_image
)

# Utility module used only to emit an audible signal
# when execution finishes or fails
from models import sound_test_finalizado


# =========================================================
# IMAGE LOADING AND PREPROCESSING
# =========================================================

def load_imgs(path_data, images_labels, target_size=(224, 224)):
    """
    Load and preprocess images using the standardized
    loader provided by grad_cam_lib.

    Parameters
    ----------
    path_data : str
        Base directory where images are stored.
    images_labels : list of str
        Relative paths to images (class_name/image.png).
    target_size : tuple
        Spatial size expected by the model.

    Returns
    -------
    list
        List of preprocessed image tensors.
    """
    images = []

    for rel_path in images_labels:
        img_path = os.path.join(path_data, rel_path)
        print(f"[INFO] loading image: {img_path}")

        images.append(
            cam.load_img_gen(
                img_path,
                target_size,
                verbose=0
            )
        )

    return images


def extract_classes(file_paths):
    """
    Extract the ground-truth class name from each image path.

    Assumes the directory structure:
        class_name/image.png

    Parameters
    ----------
    file_paths : list of str

    Returns
    -------
    list of str
        True class labels.
    """
    return [path.split("/")[0] for path in file_paths]


# =========================================================
# INFERENCE AND CAM GENERATION
# =========================================================

def run_inference_and_cams(img, model, conv_layer_name, categories):
    """
    Perform model inference and generate multiple
    Class Activation Map (CAM) variants for a single image.

    CAM methods included:
    - Grad-CAM
    - Grad-CAM++
    - Score-CAM

    Parameters
    ----------
    img : tensor
        Preprocessed input image.
    model : tf.keras.Model
        Trained CNN model.
    conv_layer_name : str
        Name of the last convolutional layer used for CAM.
    categories : list of str
        List of class labels.

    Returns
    -------
    tuple
        (probabilities, predicted_label,
         gradcam_img, gradcam_pp_img, scorecam_img)
    """

    # Run forward pass and obtain prediction probabilities
    probs, pred_label = predict_run(img, model, categories)
    prob_values = probs[0]

    # Generate Grad-CAM visualization
    gradcam = superimpose_heatmap_on_image(
        img,
        make_gradcam_heatmap(
            img,
            model,
            conv_layer_name
        ),
        alpha=0.3
    )

    # Generate Grad-CAM++ visualization
    gradcam_pp = superimpose_heatmap_on_image(
        img,
        grad_cam_plus(
            model,
            img,
            conv_layer_name
        ),
        alpha=0.3
    )

    # Generate Score-CAM visualization
    scorecam = superimpose_heatmap_on_image(
        img,
        ScoreCam(
            model,
            img,
            conv_layer_name
        ),
        alpha=0.3
    )

    return prob_values, pred_label, gradcam, gradcam_pp, scorecam


# =========================================================
# CONFIGURATION LOADING
# =========================================================

def load_config(config_path):
    """
    Load and parse a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================================================
# MAIN PIPELINE
# =========================================================

def run(config):
    """
    Main execution pipeline.

    The algorithm iterates over a list of samples defined
    in the YAML file. Each sample specifies:
    - the fold value k
    - the relative path to the image

    The model is loaded only when k changes, avoiding
    unnecessary reloads and improving efficiency.

    All results are accumulated and visualized in a
    single consolidated Grad-CAM comparison figure.
    """

    # -----------------------------------------------------
    # Global configuration parameters
    # -----------------------------------------------------
    path_data = config["data_base_path"]
    saved_dir = config["saved_dir"]
    conv_layer_name = config["conv_layer_name"]
    img_size = config["img_size"]
    output_filename = config["output_filename"]

    model_base_path = config["model_config"]["base_path"]
    model_template = config["model_config"]["filename_template"]

    target_size = (img_size, img_size)
    os.makedirs(saved_dir, exist_ok=True)

    # -----------------------------------------------------
    # Sample processing loop
    # -----------------------------------------------------
    samples_cfg = config["samples"]

    accumulated_results = []

    current_k = None
    model = None
    categories = None

    for sample in samples_cfg:

        k = sample["k"]
        image_rel_path = sample["file"]

        print(f"\n[INFO] processing k={k} | image={image_rel_path}")

        # -------------------------------------------------
        # Load model only if fold k has changed
        # -------------------------------------------------
        if k != current_k:

            model_path = os.path.join(
                model_base_path,
                model_template.format(k=k)
            )

            print(f"[INFO] loading model: {model_path}")
            model = cam.load_model(model_path)

            # Dataset directory for this fold
            base_data_path = os.path.join(path_data, f"k{k}")

            # Class labels inferred from directory structure
            categories = sorted(os.listdir(base_data_path))

            current_k = k

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------
        images = load_imgs(
            base_data_path,
            [image_rel_path],
            target_size
        )

        true_classes = extract_classes([image_rel_path])

        # -------------------------------------------------
        # Inference and CAM computation
        # -------------------------------------------------
        for img, true_class in zip(images, true_classes):

            probs, pred, gc, gcpp, sc = run_inference_and_cams(
                img,
                model,
                conv_layer_name,
                categories
            )

            accumulated_results.append({
                "image": img,
                "true_class": true_class,
                "pred_label": pred,
                "probs": probs,
                "gradcam": gc,
                "gradcam_pp": gcpp,
                "scorecam": sc
            })

    # -----------------------------------------------------
    # Final consolidated visualization
    # -----------------------------------------------------
    print("\n[INFO] generating final consolidated Grad-CAM figure")

    fig = cam.display_cam_grid_generic(
        accumulated_results,
        categories
    )

    output_path = os.path.join(saved_dir, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] image saved: {output_path}")


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a consolidated Grad-CAM comparison figure"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run(config)

        print("[INFO] finished successfully!")
        sound_test_finalizado.beep(2)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sound_test_finalizado.beep(2, str(e))
