# Weighted improved Greedy Sampling (WiGS)

## Abstract

Active learning for regression aims to reduce labeling costs by intelligently selecting the most informative data points. A prominent active learning method, improved Greedy Sampling (iGS) by [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680), determines which points are informative by balancing the feature space and the predicted output space using a static, multiplicative approach. We hypothesize that the optimal balance between these spaces is not fixed but is instead dynamic, depending on the dataset and the stage of the active learning procedure.

This paper introduces **Weighted improved Greedy Sampling (WiGS)**, a novel and flexible framework that recasts the selection criterion as a weighted, additive combination of normalized scores from the feature and predictive output space. We investigate several strategies for determining these weights: static balances, time-decay heuristics, and, most significantly, adaptive policies learned via reinforcement learning (Multi-Armed Bandits and Soft Actor-Critic). To ensure methodological rigor, our adaptive agents are trained on a "clean" reward signal based on K-fold cross-validation RMSE, while the final models are evaluated against the baseline using the standard `FullPool_RMSE` metric for a fair comparison. The SAC agent learns a data-driven policy to balance the exploration-exploitation trade-off at each iteration based on the current state of the learning process.

This entire framework is implemented as a robust, parallelized pipeline on a SLURM cluster. We evaluate our methods on 20 benchmark and synthetic regression datasets. The results demonstrate that the flexible WiGS approach, particularly the adaptive RL methods, can outperform the original iGS, demonstrating the value of adaptively balancing exploration and exploitation throughout the learning process.

## Preliminary Results

Preliminary quantitative results can be seen in the trace plots located in `Results/images/trace_plots/`. For each metric (e.g., RMSE), the `/trace` folder contains the absolute trace plots, while the `/trace_relative_iGS` folder contains the trace plots normalized relative to the iGS baseline.

These plots show that the adaptive **WiGS** methods, particularly those guided by reinforcement learning, outperform the static iGS baseline.

## Setup

This project was developed using **Python 3.9**. A virtual environment is highly recommended.

1. **Create and Activate Environment:**
```bash
   # Using venv (recommended)
   python3 -m venv .WiGS_Env
   source ./.WiGS_Env/bin/activate

```

2. **Install Requirements:**

```bash
   pip install -r requirements.txt

```

## Automated Workflow on an HPC Cluster

The project is designed as an automated pipeline for a SLURM-based HPC cluster. The scripts in `Code/Cluster/` are numbered in their execution order.

**Note:** Before running, you may need to configure cluster-specific parameters (like partition names or the number of replications) in `Code/Cluster/CreateSimulationSbatch.py` and all `.sbatch` files.

1. `0_PreprocessData.sh`: Executes `utils/Auxiliary/PreprocessData.py` to download, generate, and clean all 20 datasets, saving them as `.pkl` files in `Data/processed/`.
2. `1_CreateSimulationSbatch.sh`: Runs `CreateSimulationSbatch.py` to create all necessary `Results/` sub-directories and generate the master `.sbatch` files (one for each dataset) in `Code/Cluster/RunSimulations/`.
3. `2_RunAllSimulations.sh`: Submits all the `master_job_*.sbatch` files to the SLURM scheduler. Each of these is a job array that runs one job for every *seed* (e.g., 100 replications). This is the main experiment and populates the `Results/simulation_results/raw/` directory.
4. `3_ProcessResults.sh`: **Run this after all simulations complete.** It executes `utils/Auxiliary/AggregateResults.py` to collect all raw `.pkl` files and save them into clean, aggregated `.csv` and `.pkl` files in `Results/simulation_results/aggregated/`.
5. `4_GenerateTracePlots.sbatch`: A parallel SLURM job array that runs *per dataset*. This single script efficiently generates three sets of results:
* **All Trace Plots** (RMSE, MAE, R2, CC) for the dataset, saving to `Results/images/trace_plots/`.
* **All Weight Trends** (average and individual) for the adaptive agents, saving to `Results/images/manuscript/average_weight_trends/` and `Results/images/appendices/individual_weight_trends/`.
* **The Wilcoxon Test Table** (`.tex` file) for the dataset, saving to `Results/tables/`.


6. `5_GenerateManuscriptPlots.sh`: A local script that generates the figures in our manuscript. This is run once, after `3_ProcessResults.sh`. It creates:
* The **Nearest Neighbor** conceptual visualization.
* The **DGP plots** (two-regime and three-regime).
* The **Weight Heatmaps** (average and individual) for the synthetic datasets.
* The standalone **Legend** for the trace plots.


7. `6_GenerateSelectionVideos.sbatch`: A 2D SLURM job array that generates the `.mp4` video visualizations for each selector on the synthetic datasets. Saves to `Results/selection_videos/`.
8. `7_DeleteAuxiliaryFiles.sh` & `8_DeleteRawResults.sh`: Optional cleanup scripts to remove temporary SLURM logs and the large `raw/` simulation data.

## Directory Structure

- **`Code/`**: All executable code.
  - `Cluster/`: SLURM workflow scripts (`.sh`, `.sbatch`).
    - `RunSimulations/`: Holds generated `.sbatch` files and SLURM logs.
  - `utils/`: Core Python package.
    - `Auxiliary/`: Helper scripts (preprocessing, aggregation, plotting, visualization).
    - `Main/`: Main simulation engine (`LearningProcedure.py`).
    - `Prediction/`: ML model wrappers (`RidgeRegressionPredictor.py`) and error calculation.
    - `Selector/`: Active learning strategies (Random, GSx, iGS, WiGS variants).
- **`Data/`**:
  - `processed/`: Preprocessed `.pkl` datasets.
- **`Results/`**: All simulation outputs.
  - `images/`: All visual outputs.
    - `appendices/`: Supporting figures for the appendix.
      - `individual_weight_trends/`: Per-seed *w<sub>x</sub>* trend plots.
    - `manuscript/`: The 5-6 core figures for the paper (DGPs, heatmaps, legend).
      - `average_weight_trends/`: Average *w<sub>x</sub>* trend plots for all datasets.
    - `trace_plots/`: Trace plots organized by evaluation metric.
      - `CC/`, `MAE/`, `R2/`, `RMSE/`: Folders for each metric.
        - `trace/`: Absolute trace plots and variance.
        - `trace_relative_iGS/`: Trace plots normalized relative to the iGS baseline.
  - `simulation_results/`: Numerical data.
    - `aggregated/`: Cleaned, aggregated data organized by dataset.
      - `[Dataset_Name]/`: (e.g., `dgp_three_regime`, `dgp_two_regime`)
        - `full_pool_metrics/`: Evolution of accuracy metrics on the full pool.
        - `selection_history/`: Indices of points selected at each step.
        - `weight_history/`: Evolution of adaptive weights (*w<sub>x</sub>*) over time.
  - `tables/`: Final LaTeX tables for the Wilcoxon test results.



## Code Overview

### Main Simulation Engine (`Code/utils/Main/`)

* `LearningProcedure.py`: The core active learning loop. It trains a model, calculates both evaluation metrics, selects a point, and updates the datasets.
* `RunSimulationFunction.py`: A wrapper that runs *all* selector strategies (Passive, iGS, WiGS-SAC, etc.) for a single seed.
* `OneIterationFunction.py`: Sets up the data (loading, initial train/candidate split) for a single strategy run and calls `LearningProcedure`.
* `TrainCandidateSplit.py`: A helper script that performs the initial split between the training and candidate datasets.

### Prediction & Evaluation (`Code/utils/Prediction/`)

* `RidgeRegressionPredictor.py`, `LinearRegressionPredictor`, `RandomForestRegressorPredictor`: Wrappers for the `sklearn` models, providing `.fit()` and `.predict()` methods
* `FullPoolError.py`: Calculates the evaluation metric (RMSE, R2, etc.) based on the [iGS (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680) paper's "hybrid" method.
* `CrossValidation.py`: Calculates the RL reward signal. It gets a data-efficient and stable K-fold `CV_RMSE` using only the labeled training set (*D<sub>tr</sub>*) to prevent data leakage.

### Selector Strategies (`Code/utils/Selector/`)

* `PassiveLearningSelector.py`: Randomly samples a point (baseline).
* `GreedySamplingSelector.py`: Implements the `GSx`, `GSy`, and `iGS` baselines from [Wu et al. (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0020025518307680).
* `QBCSelector.py`: Implements a Query By Committee (QBC) strategy. It maintains a committee of 5 Ridge Regression models, each trained on a unique bootstrap sample of the current training set, and selects the candidate point with the highest prediction variance (disagreement) among the committee.
* `WeightedGreedySamplingSelector.py`: Implements `WiGS` with static and time-decaying weight heuristics.
* `WiGS_MAB.py`: Implements `WiGS` with a Multi-Armed Bandit (UCB1) that learns the best average *w<sub>x</sub>* from the `CV_RMSE` reward.
* `WiGS_SAC.py`: Implements `WiGS` with a Soft Actor-Critic (SAC) agent that learns a *state-dependent policy* to choose the optimal *w<sub>x</sub>* at each step, based on the `CV_RMSE` reward and current state.

### Auxiliary & Plotting (`Code/utils/Auxiliary/`)

- `AggregateResults.py`: Reads all raw `.pkl` files and combines them into aggregated `.csv` and `.pkl` files.
- `AnalyzeWeightTrends.py`: Generates plots showing the *w<sub>x</sub>* weight over time, supporting both average ("all") and single-seed plots.
- `DataFrameUtils.py`: A helper utility to split a pandas DataFrame into features ($X$) and the target variable ($y$).
- `GenerateAUCTable.py`: Calculates the Area Under the Curve (AUC) for the performance metrics of all selectors across all datasets. Generates a summary heatmap visualizing the relative performance of each method compared to the iGS baseline.
- `GenerateDGPImage.py`: Script to generate the specific Data Generating Process (DGP) figures for the manuscript.
- `GenerateDataTable.py`: Scans all processed datasets and generates a LaTeX table (DatasetTable.tex) summarizing their key properties, including source, sample size, and feature count.
- `GenerateJobs.py`: A helper script that generates the master SLURM .sbatch files needed to run parallel job arrays on the cluster. It configures job parameters like partition, memory, and array size based on the dataset and number of replications.
- `GeneratePerformanceTable.py`: Performs a statistical comparison (Wilcoxon signed-rank test) between specific selectors (WiGS vs. iGS, QBC vs. iGS, WiGS vs. QBC) on the synthetic datasets and prints a summary table showing significance categories and percentage improvement.
- `GeneratePlots.py`: Generates all trace plots for a given dataset. Also has a `--legend_only` mode to create the standalone legend.
- `LoadDataSet.py`: A robust utility that searches for and loads the pre-processed .pkl datasets, designed to handle differing file paths whether running locally or on the cluster.
- `NearestNeighborVisualization.py`: Script to generate the conceptual nearest-neighbor visualization for the manuscript.
- `PlotWeightHeatmap.py`: A dual-mode script that generates heatmaps.
  - `--seed 0` (or any int): Plots the heatmap for a single seed.
  - `--seed avg`: Plots the heatmap of the average *w<sub>x</sub>* across all seeds.
- `PreprocessData.py`: Downloads, generates, and cleans all 20 datasets, saving them as `.pkl` files in `Data/processed/`.
- `VerifyEndpoints.py`: A quality assurance script that verifies if all selectors start (initial pool) and end (final pool) at the exact same metric values, ensuring fair comparisons and data consistency.
- `VisualizeSelections.py`: Generates all the `.png` frames and compiles them into a final `.mp4` video.
- `WilcoxonRankSignedTest.py`: Runs a pairwise Wilcoxon signed-rank test on the aggregated results and saves a publication-ready `.tex` table.

```

```
