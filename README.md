# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Áron Bertalan
- **Aiming for +1 Mark**: No

### Solution Description

**Problem**: Automated detection of bull and bear flag patterns (normal, pennant, wedge subtypes) in forex and gold OHLC candlestick data.

**Data**: Label Studio annotations with 7 classes (no_flag + 6 pattern types). Preprocessing includes pole start standardization via slope maximization, label length filtering (16-64 bars), and file-level train/test split (75/25) to prevent data leakage from overlapping windows.

**Model**: 1D CNN with 3 convolutional blocks (16→32→64 channels), batch normalization, max pooling, dropout (0.3), and global average pooling. Input: 64-bar windows with 5 features per timestep (log returns, body ratio, upper/lower wick ratios, range). Output: 7-class softmax.

**Training**: 5-fold cross-validation with class weighting (inverse frequency), early stopping (patience=5), Adam optimizer (lr=0.001), batch size 32. Sliding window generation with stride=8, negative:positive ratio=2:1 for class balance.

**Inference**: Sliding window predictions with confidence thresholding (0.5) and overlapping detection merging. Top 10 patterns visualized with candlestick charts and confidence heatmaps.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. 

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run dl-project > log/run.log 2>&1
```

*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - `model.py`: Contains the model code (reused for training, evaluation and inference).

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: This script is used by the Docker image and local testing to execute the main pipeline stages in sequence for demonstration purposes.
