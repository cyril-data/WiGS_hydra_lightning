### Libraries ###
from sklearn.model_selection import train_test_split
import pandas as pd


### Function ###
def TrainCandidateSplit(X, y, CandidateProportion):
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
        X, y, test_size=CandidateProportion
    )
    df_Train = pd.concat([y_Train, X_Train], axis=1)

    # df_Candidate = X_Candidate.copy()
    # df_Candidate.insert(0, "Y", y_Candidate)
    df_Candidate = pd.concat([y_Candidate, X_Candidate], axis=1)

    return df_Train, df_Candidate


def TrainCandidateSplit_X(X, CandidateProportion):
    """
    Splits the original dataframe df into two sets: the training and candidate sets for features (X) only.

    Args:
        - df: The original dataframe (features only, no target column).
        - CandidateProportion: Proportion of the data to use for the candidate set.

    Returns:
        - df_Train: The initial training set (features only).
        - df_Candidate: The candidate set (features only).
    """

    # Train/Candidate split for X only
    X_Train, X_Candidate = train_test_split(X, test_size=CandidateProportion)

    return X_Train, X_Candidate
