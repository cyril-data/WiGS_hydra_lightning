# Weighted improved Greedy Sampling (WiGS) for deep learning

## Setup

This project was developed using **Python 3.9**. A virtual environment is highly recommended.

1. **Create and Activate Environment:**

With python >= 3.11 :  

```bash
   pip install -r requirements.txt
```

Create your hydra-ligtning experiment in `../henrihost-al` (see https://github.com/KarpRom/lightning-hydra-template for intel)


in `../henrihost-al`, upddate your lib : 
```bash
   pip install -r requirements.txt
```
and go back `../WiGS_hydra_lightning/Code`. 


## Simplified Workflow for deep learning


### Active learning training

In `../WiGS_hydra_lightning/Code`, run the active learning experiment : 

```bash
python RunSimulation.py \
--Data $DATA \
--TaskID $TASKID \
--NReplications $NREPLICATIONS \
--CandidateProportion $CANDIDATEPROPORTION \
--hl_xp $HL_XP \
--strat $STRAT \
--k_top $K_TOP \
--res_freq $RES_FREQ \
--hl_max_epoch $HL_MAX_EPOCH \
--hl_worker $CPU_PER_TASK
```

The args means : 
```bash
--Data 'hydralightning'             # Data from the hydralightning dataloader
--TaskID 0                          # Seed for selecting candidates
                                    # (Does not modify the initial split between trainset / candidates)
--NReplications 25                  # replication_seed = args.TaskID - 1 % args.NReplications
--CandidateProportion 0.95          # Proportion of candidates (0.95 => 5% initial trainset)
--hl_xp 100k_active_learning_h100   # Name of the hydralightning experiment in ../henrihost-al/
                                    # relative to the WiGS_hydra_lightning repo
--strat 'iGS'                       # Candidate selection strategy; Choices:
                                    # 'Passive Learning', 'GSx', 'GSy', 'iGS'
                                    # and 'WiGS' (but requires Cross Validation)
--no_cv                             # Removes Cross Validation on the trainset (required for RL in WiGS)
--k-top 100                         # Selection of the top k candidates
--subset_rand_candidat 1000         # For each candidate selection iteration:
                                    # Search for candidates not from all remaining candidates,
                                    # but from `subset_rand_candidat` randomly drawn
                                    # from the remaining candidate pool.
--hl_max_epoch 1                    # Redefines the number of epochs for each training
--curriculum                        # Preserves the model weights as candidates are selected.
                                    # This requires --no_cv
```

For a slurm script, please let's look at `Code/Cluster/RunSimulations/script_train.sh`. For example : 

```bash
bash Cluster/RunSimulations/script_train.sh \
--MACHINE v100 \
--TIME 2 \
--K_TOP 10000 \
--RES_FREQ 10
```


### Post process

All Results will be print in the dir `Results/simulation_results/raw` in `.pkl` format.

To aggregate Results, in `WiGS_hydra_lightning/Code` dir, please run :  
```bash
python utils/Auxiliary/AggregateResults.py
```

And to draw figures, please run in `WiGS_hydra_lightning` dir : 
```bash
bash Code/Cluster/4_local.sh
```

## Directory Structure

- **`Code/`**: All executable code.
  - `Cluster/`: SLURM workflow scripts (`.sh`, `.sbatch`).
    - `RunSimulations/`: Holds generated `.sbatch` files and SLURM logs.
  - `utils/`: Core Python package.
    - `Auxiliary/`: Helper scripts (preprocessing, aggregation, plotting, visualization).
    - `Main/`: Main simulation engine (`LearningProcedure.py`).
    - `Prediction/`: ML deep learning model wrappers (`LightHydra.py`) and error calculation.
    - `Selector/`: Active learning strategies (Random, GSx, iGS, WiGS variants).
- **`Data/`**:
  - `processed/`: data are read and process in `../henrihost-al` in hydraligtning way. 
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


# Optimisation and adaptation for deep learning

## Classical introduction & notation

From https://arxiv.org/abs/2603.10435, greedy Sampling frames the exploration-investigation tradeoff using a “furthest nearest neighbor” logic: selecting candidates that are maximally distant from the nearest labeled point in either feature (input space) or output space (target space).

Let the input space be $ \mathcal{X} $ and the output space be $ \mathcal{Y} $. At any iteration, we have a labeled training set $ D_{tr} = \{(x_i, y_i)\}_{i=1}^k $ and a large unlabeled candidate pool $ D_{cdd} = \{x_j\}_{j=k+1}^N $. A regression model $ f : \mathcal{X} \rightarrow \mathcal{Y} $ is trained on $ D_{tr} $. The goal is to select the sample $ x^* \in D_{cdd} $ that, when labeled, maximally improves the model’s predictive performance, defined practically as the reduction in generalization error across the entire domain.

For any candidate $x_n \in D_{cdd}$ and labeled sample $(x_m, y_m) \in D_{tr}$, we define the pairwise distances:

$$ 
d_x^{nm} \equiv \|x_n - x_m\| \text{  and  }  d_y^{nm} \equiv |f(x_n) - y_m| 
$$

- **Greedy Sampling on Features : GSx**, targets diversity in $ \mathcal{X} $ (exploration). GSx is model-agnostic, selecting the candidate with the maximum distance to its nearest labeled neighbor in the input space:
$ x^*_{\text{GSx}} = \arg\max_{x_n \in D_{cdd}} d_x^n $, where $ d_x^n \equiv \min_m d_x^{nm} $.

- **Greedy Sampling on the Output : GSy**,  targets diversity in $ \mathcal{Y} $ (investigation). GSy utilizes the current model $ f(\cdot) $ to select the candidate with the maximum prediction distance to the nearest known label:
$ x^*_{\text{GSy}} = \arg\max_{x_n \in \mathcal{D}_{\text{cdd}}} d_y^n $, where $ d_y^n \equiv \min_m d_y^{nm} $.

- **The improved Greedy Sampling : iGS**, approach creates a balanced strategy by combining both. The final score for a candidate $ x_n $ is the minimum of the products of its pairwise distances to each labeled point, and the selection criterion is to maximize this value:
$ x^*_{\text{iGS}} = \arg\max_{x_n \in D_{cdd}} s_{\text{iGS}}^n $, s.t. $ s_{\text{iGS}}^n = \min_m (d_x^{nm} \cdot d_y^{nm}) $.

-  **The Weighted improved Greedy Sampling : WiGS**, framework recasts the selection criterion as a flexible, additive combination of scores. First, to ensure the input-space exploration $d_x^{nm}$ and output-space investigation $d_y^{nm}$ metrics are comparable, we apply a normalization function $\phi(\cdot)$ to ensure the raw distances have comparable magnitudes.
  WiGS computes a weighted additive distance between each candidate and every point in the labeled set using a dynamic weight, $w(t)_x \in [0, 1]$. The final score for a candidate $x_n$ is the minimum of these combined scores taken over all labeled points:
  $$
  s_{\text{WiGS}}^n = \min_m \left[ w(t)_x \phi(d_x^{nm}) + (1 - w(t)_x) \phi(d_y^{nm}) \right] 
  $$
  The candidate observation with the highest score is selected for labeling:
  $ x^*_{\text{WiGS}} = \arg\max_{x_n \in D_{cdd}} s_{\text{WiGS}}^n $.
  For Reinforcement Learning (RL) $w(t)$ adaptation, RL can be Multi-Armed Bandits (WiGS-MAB) or Soft ActorCritic (WiGS-SAC). These methods impose a costly Cross Val computation on trainset $D_{tr}$.   


## Gready Sampling adaptation for big dataset 

For big dataset (ie several millions) with *multi dimensionnal* input space $ \mathcal{X} $ and output space $ \mathcal{Y} $, it is numerically very costly (or unfeaseable) to compute front-scratch pairwise distances from all candidates on all train samples (on input or output). 

Even with memory optimisation to compute distances in GPU between **batch** candidates and **chunk** of train samples, the cost in GPU/CPU time is very high.

We test several optimisation methods: 
- **K-TOP** : Instead of select the best candidate, we select simply k-top candidates : for example in *GSx* : 
  $$
  \arg\max_{k} \left( d_n^x \right)_{x_n \in D_{cdd}}$$
  => but this is not numerically sufficient, and may lead to select "clustered" candidates, which are not ideal to improve data dispersion. 

- **NO-CV** : We avoid all methods with Cross Val computation on trainset $D_{tr}$ (like WiGS-MAB or WiGS-SAC)

- *RandPreSelect* : We try to randomly pre-select candidates (for instance take 10000 random samples in Candidates pool and compute *GSx* on them) before the computations of min distance and argmax, but we never get improvement in comparision to pure random selection.

- **CURRICULUM** : For each candidate selection, instead of reaching a full convergence with models where weights (paramters) are initialized for each select iteration, we choose a curiculum way : only 1 training epoch and we keep the model weights through candidate selection.

At this stage K-TOP + NO-CV (GSx-y or iGS) + CURRICULUM are not numericaly sufficient to get acceptable computation times. 

The bottle neck is the distance computation. We must as far as possible limit the number of distance calculation.  

### Memory management for distance calculations

One way to do that is to keep in memory for each candidate his minimum distance to the current train set. It is at most a vector of size equal to the dataset.


To keep it simple, we start with GSx : 
$$
d_x^n(t) \text{ with } x_n \in D_{cdd} \text{=>  stored in memory} 
$$

For each new candidates selection $x'_n(t+1)$ such as $x_n(t+1) = x'_n(t+1) + x_n(t)$, the only distances to compute are those from these new candidates $x'_n(t+1)$, and the rest of candidates $D_{cdd} - x'_n(t+1)$  : 

$$
||x'_n(t+1), x_n||_{x_n \in D_{cdd}  - x'_n(t+1)} 
$$

So, at the $t+1$ candidate select iteration, if we select $k$ new candidates (len($ [x'_n(t+1)] $) = $k$ ), the number of distance calculations by select iteration is : 
$$
Cost_{distance} = [k * N_{tr}] \text{ instead of } [N_{cdd} * N_{tr}]$$
with $N_{cdd}, N_{tr}$ the current length of candidate $D_{cdd}$ and train set $D_{tr}$ (at the candidate select iteration $t+1$). 

For instance in the worst case where $N_{cdd} = N_{tr} = (\text{size} \mathcal{X}) / 2 $, we compute with memory management $k * (\text{size} \mathcal{X}) / 2$ distances instead of  $(\text{size} \mathcal{X})^2 / 4$.  


### Improve K-TOP selection 

As we said, straight k-top selection could lead to select "clustered" candidates, which are not ideal to improve data dispersion. 

As we reduce square complexity into linear (in terms of distance calculations), we can improve the k-top GSx selection by spliting into $p$ batch the k-top GSx candidate selection, and update the train set within the candidate select iteration (ie without any update of the model). 

The distance cost is still linear : 

$$
Cost_{distance} = [k * p * N_{tr}] 
$$

This improvement limits the case of selecting clusters of k-top candidates. 

### Adapt optimisation for IGs

In terms of distance calculations in output space, we can't keep $d_y^n$ because the model prediction on candidates $f(x_n)$ change through candidate select iterations. The memory management to reduce distance cost can not be directly applied.

But to keep the benefit of ouput space diversity in candidate selection (GSy or IGs), we propose to calculate $d_y^n$ only on the k-top GSx batch. 

In other words, we first select $x_p$ candidates on input distance  
$$
x_p = \arg\max_{p} \left( d_n^x \right)_{x_n \in D_{cdd}}
$$

On these $p$ candidates, we compute the output distance  and select $b$ with IGS distance : 
$$
\arg\max_{b} \left( d_n^y . d_n^y \right)_{x_p}
$$


The cost distance is still linear 
$$
Cost_{distance} = [k * p * b * N_{tr}] 
$$
The cost of prediction $f(x)$ on $p$ candidates is negligable. 


