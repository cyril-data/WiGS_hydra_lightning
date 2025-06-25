# Unique Rashomon Ensembled Active Learning (UNREAL)

## Abstract
[NeurIPS Paper/Presentation](https://neurips.cc/virtual/2024/98966)

Active learning is based on selecting informative data points to enhance model predictions often using uncertainty as a selection criterion. However, when ensemble models such as random forests are used, there is a risk of the ensemble containing models with poor predictive accuracy or duplicates with the same interpretation. To address these challenges, we develop a novel approach called *UNique Rashomon Ensembled Active Learning (UNREAL)* to only ensemble the distinct set of near-optimal models called the Rashomon set. By ensembling over the Rashomon set, our method accounts for noise by capturing uncertainty across diverse yet plausible explanations, thereby improving the robustness of the query selection in the active learning procedure. We extensively evaluate *UNREAL* against current active learning procedures on five benchmark datasets. We demonstrate how taking a Rashomon approach can improve not only the accuracy and rate of convergence of the active learning procedure but can also lead to improved interpretability compared to traditional approaches. 

## Setup

Python 3.9.18 was used in both the local and high-performance computing (HPC) cluster simulations. The following packages were used:

- `argparse` (version 1.1)
- `importlib` (part of Python standard library)
- `inspect` (part of Python standard library)
- `itertools` (part of Python standard library)
- `math` (part of Python standard library)
- `matplotlib` (version 3.9.2)
- `numpy` (version 1.26.2)
- `os` (part of Python standard library)
- `pickle` (part of Python standard library)
- `pandas` (version 2.1.4)
- `scikit-learn` (version 1.5.1)
- `scipy` (version 1.13.1)
- `treeFarms` (version from [GitHub](https://github.com/ubc-systopia/treeFarms))

## Run Simulations

Before running the simulation locally or using an HPC cluster, create the parameter vector in the `CreateParameterVector.ipynb` notebook. It is preferred to run the simulations on the HPC.

### Running Locally
Simulations can be ran locally in the `LocalSimulation.ipynb` notebook. The notebook will loop over each simulation version from the parameter vector and store it in `SimulationResults`. 

### Running on High-Performance Computing Clusters
This section will describe the functions used to run the simulations.

There are four main terminal functions in each of the folders Cluster dataset folders. They are numbered in the order that they should be ran to run the simulations.
1. `1_CreateSimulationSbatch.sh` creates an `.sbatch` file in the folder `RunSimulations` for each of the simulation set ups in the respective `ParameterVector` file. Note the `.sbatch` files are formatted to the University of Washington, Seattle's Department of Statistics high-performance computing cluster and may need to be edited in `CreateRunSimSbatch.py` to a user's institution.
2. `2_RunAllSimulations.sh` will submit each of the `.sbatch` file to the University of Washington, Seattle's Department of Statistics high-performance computing cluster to conduct the simulation. Optional emails for when the job starts, finishes, or incurs an error can be sent to the email address listed in `CreateRunSimSbatch`.
3. `3_ProcessSimResults.sh` will run the script `ProcessSimulationResults.py` to extract the error and time for each of the simulations grouped by the active learning strategies. It accesses each of the .pkl result files from the simulations, and places each error (time) into a row into the respective `ErrorMatrix.csv` (`TimeMatrix.csv`) file of each active learning strategy..
4. `4_DeleteSimulationFiles.sh` When conducting the simulations a large number of `.sbatch`, `.err`, `.out`, and `.pkl` files will be created. When the simulations are finished and collected after running '3_ProcessSimulations.sh', these excessive files become obsolete and may cause difficulty when uploading to a remote repository such as Github. As such `4_DeleteFiles` runs the following functions to delete these extra files when the simulation results have been processed.
    - `delete_sbatch.sh` deletes the `.sbatch` file used to run each simulation.
    - `delete_error.sh` deletes the error messages from each simulation.
    - `delete_out.sh` deletes the output messages from each simulation.
    - `delete_results.sh` deletes the unprocessed results from each simulation.

**WARNING:** Do not run `4_DeleteSimulationFiles.sh` before processing results with `3_ProcessSimulations.sh` .

Results will be stored in the Results folder under the respective dataset name and predictive model type (eg. RandomForestClassification or TreeFarms). The files
- ProcessedResults/ElapsedTime contains a `.csv` file with the run time of each of the iterations.
- ProcessedResults/ErrorVec contains a `.csv` file whose rows indicate the simulation iteration and whose columns indicate the error at each iteration of the active learning process.
- ProcessedResults/SelectionHistory contains a `.csv` file whose rows indicate the simulation iteration and whose columns indicate the index of the candidate observations which was queried at each iteration of the active learning process.
- ProcessedResults/TreeCount contains two `.csv` file for each simulation type. One `.csv` file is for the total number of trees from the TreeFarms model and the other `.csv` file contains the number of *unique* trees/classification patterns from the TreeFarms model. Note the `.csv` files for RandomForests will be empty.

## Code

**Main**

The following list contains the primary functions/scripts of the active learning process.
- `TrainTestCandidateSplit.py` splits the original dataframe df into three sets: the training, test, and candidate sets.
- `OneIterationFunction.py` runs one full iteration of the active learning process.
- `LearningProcedure.py` runs the active learning procedure by querying candidate observations from the candidata datset and adding them to the training set training dataset.
- `DataGeneratingProcess.py` generates data according to Burbidge, Rowland, King (2007). Note this is deprecated.

**Prediction**

The following list contains functions/scripts used for the predictive modelling of the dataset.
- `TreeFARMS.py` initializes and fits a TreeFARMS model.
- `TestErrorFunction.py` calculates the loss (RMSE for regression and classification error for classification) of the test set.
- `RidgeRegression.py` initializes and fits a ridge regression model.
- `RandomForest.py` initializes and fits a random forest model.
- `LinearRegression.py` initializes and fits a linear regression model.

**Selector**

The following list contains functions/scripts for the selection strategies of the active learning process.
- `TreeEnsembleQBCFunction.py` is a query-by-committee selection method for either random forest or Rashomon's TreeFarms that recommends an observation from the candidate set to be queried.
- `PassiveSampling.py` chooses an index at random from the candidate set to be queried.
- `GreedySampling.py` loads the three greedy sampling methods for regression from [Wu, Lin, and Huang (2018)](https://www.sciencedirect.com/science/article/pii/S0020025518307680). GSx samples based on the covariate space, GSy based on the output space, and iGS on both.

**Auxiliary**

The following list contain auxiliary functions/scripts used "behind-the-scenes" of the simulations.
- `CreateRunSimSbatch.py` is a script that creates an sbatch file to run the function `RunSimulation.py` for each parameter vector variation. Note the `.sbatch` files are formatted to the University of Washington, Seattle's Department of Statistics high-performance computing cluster and may need to be edited to a user's institution.
- `FilterArguments.py` inputs a long list of arguments and extracts only the arguments needed for a function. This function is used as different models and selector strategies use different arguments.
- `LoadDataSet.py` loads the pre-processed data into the simulation script.
- `MakeSavePlots.py` saves the matrices `MeanPlot.png` and `VariancePlot.png` into the respective `MeetingUpdates` folder.
- `MeanVariancePlot.py` creates a plot for the average error and the average variance of each active learning strategy averaged  across simulations.
- `PlotDecisionJsonTree.py`
- `ProcessSimulationResults.py` is a python script to extract the error and time for the active learning simulation. It accesses each of the .pkl result files from the simulations, and places each error (time) into a row in the `ErrorMatrix.csv` (`TimeMatrix.csv`) file.
- `WilcoxonRankSignedTest.py` computes the Wilcoxon Ranked Signed Test pairwisely for each of the methods in the simulation.
- `CreateParameterVector.ipynb` is a notebook that creates the parameter vector for each dataset.

## Analyzing Simulations
The `Analysis` folder is where the researcher can evaluate the different active learning processes. Note that if the user ran the simulations locally, they will have to manually save the files accordingly.

The first part of the notebook presents the average and maximum run time of each simulations. The second part of the notebook presents the standard active learning error plot. The final part of the notebook presents the selection history and the average iteration in which each observation was queried in the active learning process.