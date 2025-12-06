# Deep Learning Class (VITMMA19) Project Work template

[Complete the missing parts and delete the instruction parts before uploading.]

## Submission Instructions

[Delete this entire section after reading and following the instructions.]

### Project Levels

**Basic Level (for signature)**
*   Containerization
*   Data acquisition and analysis
*   Data preparation
*   Baseline (reference) model
*   Model development
*   Basic evaluation

**Outstanding Level (aiming for +1 mark)**
*   Containerization
*   Data acquisition and analysis
*   Data cleansing and preparation
*   Defining evaluation criteria
*   Baseline (reference) model
*   Incremental model development
*   Advanced evaluation
*   ML as a service (backend) with GUI frontend
*   Creative ideas, well-developed solutions, and exceptional performance can also earn an extra grade (+1 mark).

### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

[Describe the data preparation process here]

### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [x] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Áron Bertalan
- **Aiming for +1 Mark**: No

### Solution Description

[Provide a short textual description of the solution here. Explain the problem, the model architecture chosen, the training methodology, and the results.]

### Extra Credit Justification

[If you selected "Yes" for Aiming for +1 Mark, describe here which specific part of your work (e.g., innovative model architecture, extensive experimentation, exceptional performance) you believe deserves an extra mark.]

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
