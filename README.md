# Project Overview

This project is designed for comprehensive data analysis and machine learning, utilizing two main datasets: `pathmnist.npz` and `pneumoniamnist.npz`. It features two distinct models, each in their dedicated folders, A and B.

## Project Structure

- `main.py`: Orchestrates the overall workflow, handling training and evaluation for both models.
- `requirements.txt`: Lists necessary Python packages.

### Datasets

- `Datasets/`: Contains `pathmnist.npz` and `pneumoniamnist.npz`.

### Folder A

- `train.py`, `evaluate.py`, `model.py`, `utils.py`: Specific scripts for Model A operations.
- `model.pth`: Pre-trained Model A.

### Folder B

- `train.py`, `evaluate.py`, `model.py`, `utils.py`: Scripts for Model B operations.
- `pathmnist_model.pth`: Pre-trained Model B.

### Analysis Scripts

- `analysisA.py`: Performs evaluation and analysis for Model A using `pneumoniamnist.npz`.
- `analysisB.py`: Evaluates Model B using `pathmnist.npz` and includes visualization of results.

## Setup and Installation

Python 3.x required. Install dependencies via:

pip install -r requirements.txt

## Execution

To run the project, execute the `main.py` script:

python main.py

This will automatically train and evaluate both models based on the respective data in the `Datasets` folder.

## Requirements

pip install -r requirements.txt