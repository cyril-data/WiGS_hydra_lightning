# Summary: Runs active learning procedure by querying candidate observations from df_Candidate and adding them to the training set df_Train.
# Input: A dictionary SimulationConfigInputUpdated with the following keys and values:
#   DataFileInput: A string that indicates the name of the DataFrame in the Data folder.
#   df_Train: The given training dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Test: The given test dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   df_Candidate: The given candidate dataset from the function TrainTestCandidateSplit in the script OneIterationFunction.
#   Seed: Seed for reproducibility.
#   TestProportion: Proportion of the data that is reserved for testing.
#   CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
#   SelectorType: Selector mechanism in the active learning framework.
#   ModelType: Predictive model.
#   Model: The current model in the current active learning iteration.
# Output:
#   ErrorVec: A 1xM vector of errors with M being the number of observations in df_Candidate. 
#   SelectedObservationHistory: The index of the queried candidate observation at each iteration

### Import functions ###
import inspect 
import numpy as np
import pandas as pd

### Functions ###
from utils.Selector import *
from utils.Prediction import *

### Function ###
def LearningProcedure(SimulationConfigInputUpdated):

    ### Set Up ###
    i = 0
    ErrorVec = []
    SelectedObservationHistory = []

    ### Initialize Mdodel ###
    ModelClass = globals().get(SimulationConfigInputUpdated["ModelType"], None)       # Initialize the model instance
    model_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()          # Extract only relevant args for the ModelClass
                       if k in inspect.signature(ModelClass.__init__).parameters}
    predictor_model = ModelClass(**model_init_args)                                   # Create the model instance
    SimulationConfigInputUpdated['Model'] = predictor_model                           # Store this instance of the model

    ### Algorithm ###
    while len(SimulationConfigInputUpdated["df_Candidate"]) > 0:

        ## Print iteration ##
        # print(f"Iteration: {i}")

        ## Get features and target for the current training set ##
        X_train_df, y_train_series = get_features_and_target(
            df=SimulationConfigInputUpdated["df_Train"],
            target_column_name="Y",
            auxiliary_columns=SimulationConfigInputUpdated.get('auxiliary_data_cols', []))

        ## Prediction Model ##                                          
        predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)
        
        ### Test Error ###
        TestErrorOutput = TestErrorFunction(InputModel=predictor_model,
                                            df_Test=SimulationConfigInputUpdated["df_Test"])
        
        ## Store Errors ##
        ErrorVec.append(TestErrorOutput["ErrorVal"] )

        ## Sampling Procedure ##
        SelectorClass = globals().get(SimulationConfigInputUpdated["SelectorType"], None)   # Initialize the selector instance
        selector_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()         # Extract only relevant args for the SelectorClass
                              if k in inspect.signature(SelectorClass.__init__).parameters}
        selector_model = SelectorClass(**selector_init_args)                                # Create the selector instance
        SelectorFuncOutput = selector_model.select(                                         # Call the 'select' method on the selector instance
            df_Candidate=SimulationConfigInputUpdated["df_Candidate"],
            df_Train=SimulationConfigInputUpdated["df_Train"], 
            Model=predictor_model)

        ## Query selected observation ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex)

        ## Update Train and Candidate Sets ##
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex)
        
        ## Increase iteration ##
        i+=1 

    ### Output ###
    LearningProcedureOutput = {"ErrorVec": ErrorVec,
                               "SelectedObservationHistory": SelectedObservationHistory}
                              
    return LearningProcedureOutput