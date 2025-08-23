# Weighted Improved Greedy Sampling

🚧 This repository is under construction. 🚧

## Abstract

Active learning for regression aims to reduce labeling costs by intelligently selecting the most informative data points. The state-of-the-art iGS method from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680) combines input-space diversity (exploration) and output-space uncertainty (exploitation) using a multiplicative approach. This project introduces a novel, more flexible methodology called **Weighted improved Greedy Sampling (WiGS)**, which hypothesizes that the relative importance of exploration and exploitation is not equal and may change depending on the dataset and the stage of learning.

Our framework recasts the selection criterion as a weighted, additive combination of normalized diversity and uncertainty scores. We explore several strategies for determining these weights, from static, user-defined balances to adaptive heuristics like time-based decay. Most significantly, we formulate the weight selection problem within a **Multi-Armed Bandit (MAB)** framework, allowing an agent to learn an optimal, data-driven policy for balancing the exploration-exploitation trade-off at each iteration. This entire workflow is implemented in a robust, parallelized framework and evaluated on over a dozen benchmark regression datasets. The results demonstrate that the flexible WiGS approach, particularly the adaptive methods, can outperform the original iGS, demonstrating the value of adaptively balancing exploration and exploitation.

## Setup

This project was developed using **Python 3.9**. To ensure a clean and reproducible environment, it is highly recommended to use a virtual environment manager like Conda.

1.  **Create and Activate a New Conda Environment:**
    ```bash
    # Create a new environment named 'wgs_env' with Python 3.9
    conda create --name wgs_env python=3.9

    # Activate the new environment
    conda activate wgs_env
    ```

2.  **Install All Required Packages:**
    With the environment activated, install all necessary libraries using the provided `requirements.txt` file. This command will automatically install the exact versions of all packages used in this project.
    ```bash
    pip install -r requirements.txt
    ```

You are now ready to run the project's scripts.

## Automated Workflow on an HPC Cluster

The project is designed to be run as an automated pipeline on a SLURM-based HPC cluster. The scripts for managing the workflow are located in `Code/Cluster/` and are numbered in the order they should be run.

1.  `0_PreprocessData.sh`: This script executes a Python script that downloads all 15+ benchmark datasets from their sources (UCI, Kaggle Hub, StatLib/pmlb), preprocesses them into a clean format, and saves them as `.pkl` files in the `Data/processed/` directory. 

2.  `1_CreateSimulationSbatch.py`: This Python script automatically discovers all processed datasets and generates a master job script (e.g., `master_job_LinearRegressionPredictor.sbatch`) for each machine learning model you wish to test. Each master script uses a **SLURM job array** to parallelize the simulation across all datasets and all `N` replications. 

**Note:** The user can edit to the appropriate parition name (amongst other cluster inputs) in the function `create_master_sbatch` found in the file `utils/Auxiliary/GenerateJobs.py`.

3.  `2_RunAllSimulations.sh`: This shell script finds and submits all the generated master jobs to the SLURM scheduler.

4.  `3_ProcessResults.sh`: **Run this after all cluster jobs are complete.** This script executes the Python aggregation logic, which finds all the raw, individual result files from the parallel jobs and compiles them into smaller, organized `.pkl` files, saved in `Results/simulation_results/aggregated/`.

5.  `4_ImageGeneration.sh`: This script runs the final analysis, loading the aggregated results and generating all trace and variance plots for every metric (RMSE, MAE, R², CC) and saves them to the `Results/images/` directory.

6.  `5_DeleteSimulationAuxiliaryFiles.sh` & `6_DeleteRawResults.sh`: Optional cleanup scripts to remove temporary SLURM log files, sbatch scripts, and the large number of raw `.pkl` files after the analysis is complete.

## Results Structure

* **Raw Results**: Saved by cluster jobs in `Results/simulation_results/raw/{dataset_name}/{dataset}_{model}_seed_{i}.pkl`.
* **Aggregated Results**: Created by the processing script in `Results/simulation_results/aggregated/{dataset_name}/{metric}.pkl`. These are the primary files used for analysis.
* **Plots**: Saved in a nested structure in `Results/images/{metric}/{plot_type}/{plot_name}.png`.

## Code

The core logic is contained in the `Code/utils/` package, organized into sub-packages.

#### Main
* `RunSimulationFunction.py`: The main engine that, for a single seed, runs all active learning strategies on a given dataset.
* `OneIterationFunction.py`: Orchestrates a single, complete active learning run for one strategy.
* `LearningProcedure.py`: The innermost loop that iteratively trains the model, selects a sample, and records performance.
* `TrainTestCandidateSplit.py`: Splits data into initial train, test, and candidate sets.

#### Prediction
* `LinearRegressionPredictor.py`, `RidgeRegressionPredictor.py`, `RandomForestRegressorPredictor.py`: Wrapper classes for scikit-learn models.
* `TestErrorFunction.py`: Calculates standard performance metrics (RMSE, MAE, R², CC) on a held-out test set.
<!-- * [cite_start]`PaperEvaluation.py`: Calculates performance using the specific "in-pool" metric described in the original Wu, Lin, and Huang (2018) paper[cite: 185]. -->

#### Selector
* `PassiveLearningSelector.py`: Implements random sampling (baseline).
* `GreedySamplingSelector.py`: Implements the original GSx, GSy, and iGS methods from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680).
* `WeightedGreedySamplingSelector.py`: Implements the novel **WiGS** method with static and time-decay adaptive weights.
* `WiGS_MAB.py`: Implements the novel **WiGS** method using a Multi-Armed Bandit (UCB1) to learn weights adaptively.

#### Auxiliary
* `preprocess_data.py`: The script for downloading and cleaning all datasets.
* `GenerateJobs.py`: The library function for creating the content of the `master_job.sbatch` files.
* `aggregate_results.py`: The library script for compiling raw parallel results into aggregated files.
* `GeneratePlots.py`: The library script containing the main plotting logic.
* `MeanVariancePlot.py`: The underlying function that creates the trace and variance plot figures.