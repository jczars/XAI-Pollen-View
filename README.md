# XAI-Pollen-View
This repository contains the implementation of XAI-Pollen-View, a framework designed to interpret Deep Neural Network decisions in pollen classification. We compare activation mappings between polar coordinate transformations and original image formats to identify which features most influence the model's accuracy.

**Phase 1:** Refinement of selected models and classification of datasets.

**Phase 2:** Separation of the test set into views (Equatorial and Polar) using pseudo-labeling.

**Phase 3:** Classification of the test set and evaluation of classification metrics.

To quickly run all tests, follow this menu:

**To run all the tests quickly, follow this menu:**

## Table of Contents

- [Pollen_classification_CBAM](#XAI-Pollen-View)
  - [Installation](#installation)
- [Usage](#usage)
  - [Phase 1](#phase-1)
  - [Phase 2](#phase-2)
  - [Phase 3](#phase-3)
- [Results](#Results)
  - [Results of Phase 3](#Results-of-dataset-separation-by-views)
- [Discussion](#Discussion)
  - [Consolidated results](#Consolidated-results)
  - [Test the Wilcoxon](#Test-wilcoxon)
  - [Compare metrics](#compare-metrics)
  - [Interpretability](#Interpretability)
- [Project Folder Structure](#project-Folder-Structure)
  - [Description of Key Folders](#description-of-Key-Folders)
  - [Resources](#resources)

# Installation

Follow the steps below to set up the project environment and dependencies.


**1. Create and activate the virtual environment**
There are several ways to create a virtual environment. In this project, the conda package was used, but you can also choose other tools such as venv or virtualenv based on your preference.

To create the virtual environment using conda, run the following command:
```bash
conda create --name tfGpu python=3.10.13
```

**Activate the virtual environment**:
```bash
conda activate tfGpu
```

**2. Clone the repository**
```bash
git clone https://github.com/jczars/XAI-Pollen-View.git
```
**3. Install dependencies**
```bash
cd XAI-Pollen-View/
pip install -r requirements.txt
```

**4. Verify the Installation**
After installing the dependencies, you can check if everything was set up correctly. Use the following commands to check the installed packages and the Python version:
```bash
pip list
python3 --version

```
5. Deactivate the Virtual Environment
Once you’re done working on the project, deactivate the virtual environment with the command:
```bash
conda deactivate
```
This will return you to the global system environment.

**6. Re-activate the Virtual Environment**
Whenever you continue working on the project, remember to reactivate the virtual environment:
```bash
conda activate tfGpu
```
By following these steps, you’ll have an isolated environment for the project using conda to manage dependencies and avoid conflicts with other Python installations on your system.

**7. Navigate to the project directory**
```bash
cd XAI-Pollen-View
```

**8. Adjust the Python Path (if needed)**
If you encounter issues with module imports, you can manually adjust the `PYTHONPATH`:

Find the current directory:
Run the following command in the terminal to get the current working directory:

To include the project path:
```bash
pwd
```
Build the export PYTHONPATH command:
Combine the result of pwd with the rest of the command to set PYTHONPATH. Assuming the current directory is the desired one:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

To remove the project path:
```bash
unset PYTHONPATH
```



