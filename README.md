# Weighted Improved Greedy Sampling

## Abstract

Active learning for regression aims to reduce labeling costs by intelligently selecting the most informative data points. The state-of-the-art iGS method from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680) combines input-space diversity (exploration) and output-space uncertainty (exploitation) using a multiplicative approach. This project introduces a novel, more flexible methodology called **Weighted improved Greedy Sampling (WiGS)**, which hypothesizes that the relative importance of exploration and exploitation is not equal and may change depending on the dataset and the stage of learning.

Our framework recasts the selection criterion as a weighted, additive combination of normalized diversity and uncertainty scores. We explore several strategies for determining these weights: static balances, time-based decay heuristics, and, most significantly, adaptive policies learned via reinforcement learning (Multi-Armed Bandits and Soft Actor-Critic). The SAC agent learns a data-driven policy to balance the exploration-exploitation trade-off at each iteration based on the current state of the learning process. This entire workflow is implemented in a robust, parallelized framework using SLURM and evaluated on over 20 benchmark and synthetic regression datasets.

The results demonstrate that the flexible WiGS approach, particularly the adaptive RL methods, can outperform the original iGS, demonstrating the value of adaptively balancing exploration and exploitation throughout the learning process.

## Preliminary Results

Preliminary quantitative results can be seen in the trace plots located in `Results/images/full_pool/RMSE`. The folder [`/trace`](https://github.com/thatswhatsimonsaid/WeightedGreedySampling/tree/a6ba77f8ab02da6166411e08d350926344d4082d/Results/images/full_pool/RMSE/trace/trace) contains the typical trace plots, while [`/trace_relative_iGS`](https://github.com/thatswhatsimonsaid/WeightedGreedySampling/tree/a6ba77f8ab02da6166411e08d350926344d4082d/Results/images/full_pool/RMSE/trace_relative_iGS/trace) contains the trace plot relative to [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680)'s iGS method.

These plots show that the adaptive **WiGS** methods, particularly those guided by reinforcement learning methods, generally outperform the static iGS baseline.

A visualization demonstrating the adaptive behavior of the WiGS (SAC) agent on the dgp_three_regime dataset is shown below. Observe how the agent adjusts its exploration/exploitation strategy (indicated by the $w_x^{(t)}$ weight) throughout the learning process.

<div align="center">

![WiGS SAC Demo](./Resu;ts/images/Presentation/wigs_sac_demo.gif)

</div>

## Setup

This project was developed using **Python 3.9**. A virtual environment is highly recommended.

1.  **Create and Activate Environment:**
    ```bash
    # Using venv (recommended)
    python3 -m venv .WiGS_Env
    source ./.WiGS_Env/bin/activate

    # Or using Conda
    # conda create -n WiGS_Env python=3.9
    # conda activate WiGS_Env
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Automated Workflow on an HPC Cluster

The project is designed as an automated pipeline for a SLURM-based HPC cluster. The scripts in `Code/Cluster/` are numbered in their execution order.

**Note:** The user can edit to the appropriate parition name (amongst other cluster inputs) in the file `Code/Cluster/CreateSimulationSbatch.py`, `5_ImageGeneration.sbatch`, and `6_CompileAllPlots.sbatch`.


1.  `1_PreprocessData.sh`: This script executes a Python script that downloads all 20 benchmark datasets from their sources (UCI, Kaggle Hub, StatLib/pmlb), preprocesses them into a clean format, and saves them as `.pkl` files in the `Data/processed/` directory. 

2.  `2_CreateSimulationSbatch.py`: This Python script automatically discovers all processed datasets and generates a master job script (e.g., `master_job_LinearRegressionPredictor.sbatch`) for each machine learning model you wish to test. Each master script uses a **SLURM job array** to parallelize the simulation across all datasets and all `N` replications. 

3.  `3_RunAllSimulations.sh`: Submits all generated master jobs to the SLURM scheduler.

4.  `4_ProcessResults.sh`: **Run after all cluster jobs complete.** Executes `AggregateResults.py` to collect raw `.pkl` results and save them into organized, per-dataset aggregated files:
    * Aggregated metrics (`.../full_pool_metrics/*.pkl`).
    * Selection history (`.../selection_history/*.csv`).
    * Weight history (`.../weight_history/*.csv`).
    * Initial training indices (`.../InitialIndices.csv`).
    * Elapsed times (`.../ElapsedTime.csv`).

5.  `5_ImageGeneration.sbatch`: Submits a parallel job array to generate all individual trace plots for every dataset, metric, and evaluation type. 

6. `6_GenerateDGPAndTrends.sh`: Executes `GenerateDGPImage.py` to create visualizations of the synthetic datasets (saving to `docs/images/`) and runs `VisualizeWeightTrends.py` to generate plots of the $w_x^{(t)}$ weights over iterations (saving to `Results/visualizations/`).

7. `7_GenerateVideos_bySelector.sbatch`: (🚧 Under construction 🚧) Submits a **SLURM job array** where each task generates the video frames and compiles the video (`.mp4`) for a specific selector strategy using `VisualizeSelections.py`. Outputs saved in `Results/visualizations/`.

8.  `8_DeleteAuxiliaryFiles.sh` & `9_DeleteRawResults.sh`: Optional cleanup scripts to remove temporary logs and raw data.

9.  `10_CompileAllPlots.sbatch`: Submits a parallel job array to compile the individual plots into summary grid images, perfect for presentations. This script is highly configurable for different layouts.


## Directory Structure

* **`Code/`**: All executable code.
    * `Cluster/`: SLURM workflow scripts (`.sh`, `.sbatch`, `.py`).
        * `RunSimulations/`: Holds generated `.sbatch` files and SLURM logs.
    * `Notebooks/`: Jupyter notebooks for exploration.
    * `utils/`: Core Python package.
        * `Auxiliary/`: Helper scripts (preprocessing, aggregation, plotting, visualization).
        * `Main/`: Main simulation engine (`LearningProcedure.py`).
        * `Prediction/`: ML model wrappers (`RidgeRegressionPredictor.py`) and error calculation (`FullPoolError.py`).
        * `Selector/`: Active learning strategies (Random, GSx, iGS, WiGS variants).
* **`Data/`**:
    * `processed/`: Preprocessed `.pkl` datasets.
* **`Results/`**: All simulation outputs.
    * `images/`: Plots and video frames.
        * `full_pool/`: Trace plots (absolute and relative to iGS).
        * `Presentation/`: Images used in our presentation.
        * `(Planned)` `video_frames/`: Individual frames generated by `VisualizeSelections.py`.
    * `simulation_results/`: Numerical data.
        * `raw/`: Individual `.pkl` output from each cluster job.
    * `aggregated/`: Per-dataset folders containing:
            * `full_pool_metrics/`: Aggregated metric data (`.pkl`).
            * `selection_history/`: Order of selected indices (`.csv`).
            * `weight_history/`: $w_x$ weights used by adaptive methods (`.csv`).
            * `InitialIndices.csv`: Indices of the initial training set for all seeds (`.csv`).
            * `ElapsedTime.csv`: Runtimes per strategy/seed (`.csv`).
    * `visualizations/`: Outputs from visualization scripts

## Code Overview

#### Main Functions

* `LearningProcedure.py`: The core active learning loop.
* `OneIterationFunction.py`: Manages setup (data loading/splitting) and execution for one strategy run. Calls `LearningProcedure`.
* `RunSimulationFunction.py`: Runs all selector strategies for a single dataset and seed. Called by `RunSimulation.py`.
* `TrainCandidateSplit.py`: Splits data into initial `df_Train` and `df_Candidate`.

#### Prediction Functions

* `LinearRegressionPredictor.py`, `RidgeRegressionPredictor.py`, and `RandomForestRegressorPredictor.py`: Wrappers for scikit-learn models.
* `FullPoolError.py`: Calculates performance metrics using the "Full Pool" method from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680).

#### Selector Functions
* `GreedySamplingSelector.py`: Implements GSx, GSy, and iGS [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680).
* `PassiveLearningSelector.py`: Random sampling baseline.
* `WeightedGreedySamplingSelector.py`: Implements **WiGS** with static and time-decay weights. Returns `w_x`.
* `WiGS_MAB.py`: Implements **WiGS** using a Multi-Armed Bandit (UCB1) to choose weights. Returns `w_x`.
* `WiGS_SAC.py`: Implements **WiGS** using a Soft Actor-Critic agent to learn a policy for choosing `w_x`. Returns `w_x`.

#### Auxiliary Functions

* `AggregateResults.py`: Compiles raw `.pkl` results into aggregated `.pkl` (metrics) and `.csv` (history) files.
* `AnalyzeWeightTrends.py`: Generates plots showing the trend of $w_x^{(T)}$ weights over iterations.
* `CompilePlots.py.py`: Stitches individual trace plots into combined grid images.
* `DataFrameUtils.py`: Helper function to split a DataFrame into features (X) and target (y).
* `GenerateJobs.py`: Creates the content for SLURM master job scripts.
* `GeneratePlots.py`: Creates trace plots from aggregated metric data.
* `GenerateDGPImage.py`: Creates visualizations of the synthetic DGP datasets (two-regime, three-regime).
* `LoadDataSet.py`: Loads a specific `.pkl` dataset.
* `PreprocessData.py`: Downloads, generates, cleans, and saves all datasets.
* `(Planned)` `VisualizeSelections.py`: Generates plot frames for videos from aggregated history CSVs.
