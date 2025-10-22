### Libraries ###
from sklearn.model_selection import train_test_split

### Function ###
def TrainCandidateSplit(df, CandidateProportion):
    """
    Splits the original dataframe df into two sets: the training and candidate sets.

    Args:
        - df: The original dataframe.
        - CandidateProportion: Proportion of the data that is initially "unseen" and later added to the training set.
    
    Returns:
        - df_Train: The initial training set.
        - df_Candidate: The candidate set that is initially "unseen" and later added to the training set.
    """

    # Train/Candidate split #
    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(
        df.loc[:, df.columns != "Y"], df["Y"], test_size=CandidateProportion)
    
    # Column names #
    df_Train = X_Train.copy()
    df_Train.insert(0, 'Y', y_Train)
    df_Candidate = X_Candidate.copy()
    df_Candidate.insert(0, 'Y', y_Candidate)

    return df_Train, df_Candidate