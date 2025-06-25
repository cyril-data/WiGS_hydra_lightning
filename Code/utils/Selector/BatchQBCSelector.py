# Summary: Query-by-committee selector that recommends observations from the candidate set to be queried,
#          incorporating diversity and density metrics.

### Libraries ###
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from utils.Auxiliary.DataFrameUtils import get_features_and_target

class BatchQBCSelector:

    ### Initialize ###
    def __init__(self, 
                 UniqueErrorsInput: int, 
                 DiversityWeight: float, 
                 DensityWeight: float, 
                 BatchSize: int, 
                 Seed: int = None, 
                 **kwargs):
        self.UniqueErrorsInput = UniqueErrorsInput
        self.DiversityWeight = DiversityWeight
        self.DensityWeight = DensityWeight
        self.BatchSize = BatchSize
        self.Seed = Seed 

        ### Ignore warning (taken care of) ###
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply", category=RuntimeWarning)

    ### Select ###
    def select(self, df_Candidate: pd.DataFrame, Model, df_Train: pd.DataFrame, auxiliary_columns: list = None) -> dict:
        
        ## Candidate and Training datasets Set Up ##
        X_Candidate_df, _ = get_features_and_target(
            df=df_Candidate,
            target_column_name="Y",
            auxiliary_columns=auxiliary_columns)
        
        X_Train_df, _ = get_features_and_target( 
            df=df_Train,
            target_column_name="Y", 
            auxiliary_columns=auxiliary_columns)

        ## Predicted Values (Committee Votes) ##
        preds_candidate_raw_df = Model.get_raw_ensemble_predictions(X_Candidate_df)
        preds_train_raw_df = Model.get_raw_ensemble_predictions(X_Train_df) 
        all_preds_combined_df = pd.concat([preds_candidate_raw_df, preds_train_raw_df], axis=0, ignore_index=True) 

        ## Unique Classification Patterns ##
        unique_trees_preds_combined_df = all_preds_combined_df.T.drop_duplicates().T 

        ## Initialize Output ##
        Output = {}

        ## Store tree counts ##
        if hasattr(Model, 'get_tree_counts'): 
            tree_counts = Model.get_tree_counts()
            Output["AllTreeCount"] = tree_counts.get("AllTreeCount", preds_candidate_raw_df.shape[1])
            Output["UniqueTreeCount"] = unique_trees_preds_combined_df.shape[1] # Use the shape of the derived unique patterns
        else: 
            Output["AllTreeCount"] = preds_candidate_raw_df.shape[1] # Total estimators from model
            Output["UniqueTreeCount"] = unique_trees_preds_combined_df.shape[1] # Actual unique patterns found


        ## Select Committee for Vote Entropy based on UniqueErrorsInput ##
        committee_preds_df = None
        num_estimators_for_entropy = 0

        if self.UniqueErrorsInput: # UNREAL
            committee_preds_df = preds_candidate_raw_df[unique_trees_preds_combined_df.columns] 
            num_estimators_for_entropy = committee_preds_df.shape[1]
        else: # DUREAL
            committee_preds_df = preds_candidate_raw_df
            num_estimators_for_entropy = committee_preds_df.shape[1]

        # Convert to NumPy array
        PredictedValues = committee_preds_df.values 

        # Defensive check for empty predictions
        if PredictedValues is None or PredictedValues.size == 0 or num_estimators_for_entropy == 0:
            Output["IndexRecommendation"] = []
            return Output

        ## Vote Entropy Calculation ##
        UniqueClasses = np.unique(df_Train["Y"])
        num_candidate_samples = PredictedValues.shape[0]
        VoteEntropyFinal = np.zeros(num_candidate_samples)

        for idx_sample in range(num_candidate_samples):
            sample_predictions = PredictedValues[idx_sample, :] 
            
            # Calculate vote proportions for each class for this sample #
            counts = pd.Series(sample_predictions).value_counts().reindex(UniqueClasses, fill_value=0)
            proportions = counts / num_estimators_for_entropy 
            
            # Calculate entropy for this sample #
            entropy_terms = -proportions * np.log(proportions + np.finfo(float).eps)
            VoteEntropyFinal[idx_sample] = np.sum(entropy_terms)
        
        ## Measures ##
        DiversityValues = df_Candidate["DiversityScores"]
        DensityValues = df_Candidate["DensityScores"]

        ## Normalize ##
        scaler = MinMaxScaler()
        DiversityValues_scaled = scaler.fit_transform(np.array(DiversityValues).reshape(-1, 1)).flatten()
        DensityValues_scaled = scaler.fit_transform(np.array(DensityValues).reshape(-1, 1)).flatten()
        VoteEntropyFinal_scaled = scaler.fit_transform(np.array(VoteEntropyFinal).reshape(-1, 1)).flatten()

        ### Uncertainty Metric ###
        df_Candidate["UncertaintyMetric"] = (1 - self.DiversityWeight - self.DensityWeight) * VoteEntropyFinal_scaled + \
                                            self.DiversityWeight * DiversityValues_scaled + \
                                            self.DensityWeight * DensityValues_scaled
        
        ## Ensure BatchSize doesn't exceed available candidates ##
        batch_size_actual = min(self.BatchSize, df_Candidate.shape[0])
        if batch_size_actual > 0:
            IndexRecommendation = list(df_Candidate.sort_values(by="UncertaintyMetric", ascending=False).index[0:batch_size_actual])
        else:
            IndexRecommendation = []

        ## Drop uncertainty metric from candidate data frame ##
        df_Candidate.drop('UncertaintyMetric', axis=1, inplace=True)

        ## Output ##
        Output["IndexRecommendation"] = IndexRecommendation

        return Output