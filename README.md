# Weighted Improved Greedy Sampling

🚧 This repository is under construction. 🚧

Preliminary results can be seen in `Results/images/full_pool/RMSE`. The folder [`/trace`](https://github.com/thatswhatsimonsaid/WeightedGreedySampling/tree/main/Results/images/full_pool/RMSE/trace/trace) contains the typical trace plots, while [`/trace_relative_iGS`](https://github.com/thatswhatsimonsaid/WeightedGreedySampling/tree/main/Results/images/full_pool/RMSE/trace_relative_iGS/trace) contains the trace plot relative to [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680)'s iGS method.

As shown, the adaptive **WiGS** methods, particularly those guided by reinforcement learning, generally outperform the static iGS baseline.

## Abstract

Active learning for regression aims to reduce labeling costs by intelligently selecting the most informative data points. The state-of-the-art iGS method from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680) combines input-space diversity (exploration) and output-space uncertainty (exploitation) using a multiplicative approach. This project introduces a novel, more flexible methodology called **Weighted improved Greedy Sampling (WiGS)**, which hypothesizes that the relative importance of exploration and exploitation is not equal and may change depending on the dataset and the stage of learning.

Our framework recasts the selection criterion as a weighted, additive combination of normalized diversity and uncertainty scores. We explore several strategies for determining these weights, from static, user-defined balances to adaptive heuristics like time-based decay. Most significantly, we formulate the weight selection problem within a reinforcement learning framework, allowing an agent (*MAB* or *SAC*) to learn an optimal, data-driven policy for balancing the exploration-exploitation trade-off at each iteration. This entire workflow is implemented in a robust, parallelized framework and evaluated on over a dozen benchmark regression datasets. 

The results demonstrate that the flexible WiGS approach, particularly the adaptive methods, can outperform the original iGS, demonstrating the value of adaptively balancing exploration and exploitation.

## Setup

This project was developed using **Python 3.9**. To ensure a clean and reproducible environment, it is highly recommended to use a virtual environment manager like Conda.

1.  **Create and Activate a New Conda Environment:**
    ```bash
    # Create a new environment named 'WiGS_Env'
    python3 -m venv .WiGS_Env

    # Activate the new environment
    source ./.WiGS_Env/bin/activate
    ```

2.  **Install All Required Packages:**
    With the environment activated, install all necessary libraries using the provided `requirements.txt` file. This command will automatically install the exact versions of all packages used in this project.
    ```bash
    pip install -r requirements.txt
    ```
## Automated Workflow on an HPC Cluster

The project is designed as an automated pipeline for a SLURM-based HPC cluster. The scripts in `Code/Cluster/` are numbered in their execution order.

**Note:** The user can edit to the appropriate parition name (amongst other cluster inputs) in the file `Code/Cluster/CreateSimulationSbatch.py`, `5_ImageGeneration.sbatch`, and `6_CompileAllPlots.sbatch`.


1.  `1_PreprocessData.sh`: This script executes a Python script that downloads all 20 benchmark datasets from their sources (UCI, Kaggle Hub, StatLib/pmlb), preprocesses them into a clean format, and saves them as `.pkl` files in the `Data/processed/` directory. 

2.  `2_CreateSimulationSbatch.py`: This Python script automatically discovers all processed datasets and generates a master job script (e.g., `master_job_LinearRegressionPredictor.sbatch`) for each machine learning model you wish to test. Each master script uses a **SLURM job array** to parallelize the simulation across all datasets and all `N` replications. 

3.  `3_RunAllSimulations.sh`: Submits all generated master jobs to the SLURM scheduler.

4.  `4_ProcessResults.sh`: **Run this after all cluster jobs are complete.** Aggregates the raw result files into organized, per-dataset files. Results are split by evaluation type into `test_set_metrics` and `full_pool_metrics` folders.

5.  `5_ImageGeneration.sbatch`: Submits a parallel job array to generate all individual trace plots for every dataset, metric, and evaluation type. The `--no-legend` flag can be added to this script's Python call to generate plots ready for compilation.

6.  `6_CompileAllPlots.sbatch`: Submits a parallel job array to compile the individual plots into summary grid images, perfect for presentations. This script is highly configurable for different layouts.

7.  `7_DeleteAuxiliaryFiles.sh` & `8_DeleteRawResults.sh`: Optional cleanup scripts to remove temporary logs and raw data.

## Directory Structure

* **`Code/`**: Contains all executable code.
    * `Cluster/`: Holds all shell and sbatch scripts for managing the HPC workflow.
    * `Notebooks/`: Jupyter notebooks for exploratory analysis.
    * `utils/`: The core Python package for the project.
        * `Auxiliary/`: Helper scripts for preprocessing, aggregation, and plotting.
        * `Main/`: The main simulation engine (`LearningProcedure.py`).
        * `Prediction/`: Wrappers for machine learning models and error calculation functions (`HoldOutError.py`, `FullPoolError.py`).
        * `Selector/`: Implementations of all active learning strategies (Random, GSx, iGS, WiGS, etc.).
* **`Data/`**: Stores the preprocessed `.pkl` datasets.
* **`Results/`**: Contains all outputs from the simulations.
    * `simulation_results/`: Raw and aggregated numerical data.
    * `images/`: Individual plots and final compiled grid images.

## Code

The core logic is contained in the `Code/utils/` package, organized into sub-packages.

#### Main
* `RunSimulationFunction.py`: The main engine that, for a single seed, runs all active learning strategies on a given dataset.
* `OneIterationFunction.py`: Orchestrates a single, complete active learning run for one strategy.
* `LearningProcedure.py`: The innermost loop that iteratively trains the model, selects a sample, and records performance.
* `TrainTestCandidateSplit.py`: Splits data into initial train, test, and candidate sets.

#### Prediction
* `LinearRegressionPredictor.py`, `RidgeRegressionPredictor.py`, `RandomForestRegressorPredictor.py`: Wrapper classes for scikit-learn models.
* `TestErrorFunction.py`: Calculates standard performance metrics (RMSE, MAE, R^2, CC) on a held-out test set.

#### Selector
* `PassiveLearningSelector.py`: Implements random sampling (baseline).
* `GreedySamplingSelector.py`: Implements the original GSx, GSy, and iGS methods from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680).
* `WeightedGreedySamplingSelector.py`: Implements the novel **WiGS** method with static and time-decay adaptive weights.
* `WiGS_MAB.py`: Implements the novel **WiGS** method using a Multi-Armed Bandit (UCB1) to learn weights adaptively.
* `WiGS_SAC.py`: Implements the novel **WiGS** method using a Soft Actor-Critic (SAC) to learn weights adaptively.

#### Auxiliary
* `AggregateResults.py`: The script for compiling raw parallel results into aggregated files.
* `DataFrameUtils.py`: Extracts the parameters used for specific selector and prediction methods.
* `GenerateJobs.py`: The function for creating the content of the `master_job.sbatch` files.
* `GeneratePlots.py`: The script containing the main plotting logic for the trace plots.
* `LoadDataSet.py`: Wrapper function to load a dataset.
* `PreprocessData.py`: The script for downloading and cleaning all datasets.
