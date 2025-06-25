'''
Some code based on Leveraging Predictive Equivalence in Decision Trees
'''
from treefarms import TREEFARMS as tf
import pandas as pd
import numpy as np

DEFAULT_STATIC_CONFIG = {
    "depth_budget": 3,
    "rashomon_ignore_trivial_extensions": True,
    "regularization": 0.02,
    "verbose": False
}

def _predict(tree_model, X):
    """
    A faster version of the predict function from TreeClassifier

    Parameters
    ---
    X : matrix-like, shape = [n_samples by m_features]
        a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

    Returns
    ---
    array-like, shape = [n_samples by 1] : a column where each element is the prediction associated with each row
    """
    predictions = []
    (n, m) = X.shape

    # By only accessing .values once, we avoid duplicating the
    # entire array a ton of times
    data = X.values
    for i in range(n):
        prediction, _ = tree_model.classify(data[i, :])
        predictions.append(prediction)
    return pd.Series(predictions)
def _score(tree_model, X, y):
    return (_predict(tree_model, X) == y).mean()


class Treefarms_LFR: 
    '''
    Class to allow re-use of treefarms with less frequent refits
    - instead of fitting from scratch under slight data perturbations, 
      this code will run subset operations on the set of
      previously found models.
    TODO: 
    - extend to other forms of epsilon besides additive
    - allow tree objective computations to be updated in more of a streaming style
    - formalize whether we're using score or objective for the epsilon bound
    '''
    def __init__(self, static_config: dict = DEFAULT_STATIC_CONFIG):
        self.static_config = static_config
    
    def fit(self, X: pd.DataFrame, y: pd.Series, epsilon):
        '''
        Requires: epsilon, the additive bound to use for the inital fit
        ''' 
        if epsilon < 0 or epsilon > 1: 
            raise ValueError(f"Epsilon value of {epsilon} is invalid for Rashomon sets; must be within [0, 1]")
        self.full_epsilon = epsilon
        self.epsilon = epsilon
        self.X = X
        self.y = y
        config = self.static_config.copy()
        config['rashomon_bound_adder'] = epsilon
        self.tf = tf(config)
        self.tf.fit(X, y)
        #overhead to get models
        self.all_trees = [self.tf[i] for i in range(self.tf.get_tree_count())]
        self.trees_in_scope = self.all_trees.copy()
    
    def all_predictions_one_sample(self, x: pd.Series): 
        '''
        Gives predictions for every in-scope Rset tree for sample x
        '''
        return pd.Series([tree.classify(x)[0] for tree in self.trees_in_scope])
    
    def all_predictions(self, X: pd.DataFrame): 
        '''
        Returns predictions for every tree in Rset for every 
        row of X
        output is (num rows in X) by (num trees in scope)
        '''
        return X.apply(self.all_predictions_one_sample, axis=1)

    def refit(self, X_to_add, y_to_add, epsilon, verbose=True):
        '''
        Assumes self.all_trees is length at least 1
        returns True if did fresh fit call, 
        False if filtered existing rset
        '''
        all_X = pd.concat([self.X, X_to_add], ignore_index=True)
        all_y = pd.concat([self.y, y_to_add], ignore_index=True)
        if self.full_epsilon < epsilon: #TODO: adjust this check based on robustness bounds (or relaxations thereof)
            if verbose:
                print("new epsilon larger than current epsilon, fitting fresh RSet")
            self.fit(all_X, all_y, epsilon)
            return True
        else:
            objectives = np.array([_score(tree, all_X, all_y) for tree in self.all_trees])
            indices = np.argsort(objectives)
            #remove indices with objective worse than epsilon above the minimum
            min_objective = objectives[indices[0]]
            self.trees_in_scope = [self.all_trees[i] for i in indices if objectives[i] <= min_objective + epsilon]
            self.X = all_X
            self.y = all_y
            self.epsilon = epsilon
            return False