from lfr.treefarms_wrapper import Treefarms_LFR, _score, _predict
import pandas as pd
import numpy as np
import tqdm

class tuning_Treefarms_LFR(Treefarms_LFR): 
    '''
    Class to run online TreeFarms while actively tuning epsilon as each sample is updated
    '''

    def fit(self, X: pd.DataFrame, y: pd.Series, epsilon: float):
        '''
        treats epsilon as the max epsilon; tunes with that as upper bound
        '''
        if 'verbose' in self.static_config and self.static_config['verbose']: 
            print("Calling regular Treefarms_LFR with provided maximum value of {epsilon} for epsilon")
        super().fit(X, y, epsilon)
        all_accuracies = np.array([_score(tree, X, y) for tree in self.all_trees])
        self.accuracy_ordering = np.argsort(all_accuracies)

        self.predictions = np.zeros((X.shape[0], len(self.all_trees)))
        for j in range(len(self.all_trees)): 
            self.predictions[:, j] = _predict(self.all_trees[self.accuracy_ordering[j]], X).values
        self._tune_eps(sorted_accs=all_accuracies[self.accuracy_ordering])
    
    def _tune_eps(self, sorted_accs): 
        # todo: improve efficiency of scan
        best_num_trees = 1
        best_acc = 0
        for i in range(len(self.all_trees)):
            if i < len(self.all_trees)-1 and sorted_accs[i] == sorted_accs[i+1]: 
                continue #split not valid for any epsilon threshold
            # take the i+1 best trees
            # and compute the majority vote of those trees
            y_hat = self.predictions[:,:i+1].mean(axis=1)>0.5
            acc = np.mean(y_hat == self.y)
            if acc > best_acc: 
                best_num_trees = i+1
                best_acc = acc
        
        #compute trees in scope
        self.trees_in_scope = [self.all_trees[k] for k in self.accuracy_ordering[:best_num_trees]]
        #compute epsilon 
        if best_num_trees == len(self.all_trees): 
            self.epsilon = self.full_epsilon
        else:
            # if we don't include all trees, we set epsilon to be halfway between 
            # performance of last tree in scope and 
            # performance of first tree out of scope
            lower = self.all_accuracies[best_num_trees-1]
            upper = self.all_accuracies[best_num_trees]
            self.epsilon = (upper + lower) / 2 - self.all_accuracies[0]
            
    def refit(self, X_to_add, y_to_add, epsilon, verbose=True):
        if not super().refit(X_to_add, y_to_add, epsilon, verbose=verbose): 
            # if we didn't do a fresh fit call, have to update predictions and accuracy order: 
            predictions_to_add = np.zeros((X_to_add.shape[0], self.predictions.shape[1]))
            for j in range(len(self.all_trees)): 
                predictions_to_add[:, j] = _predict(self.all_trees[self.accuracy_ordering[j]], X_to_add).values
            self.predictions = np.concatenate([self.predictions, predictions_to_add])
            
            # get accuracies for matrix and resort matrix/accuracies accordingly
            accuracies = (self.predictions == self.y.values[:, np.newaxis]).mean(axis=0) #accuracies for current accuracy ordering
            map_cur_to_new_ordering = accuracies.argsort()
            self.accuracy_ordering = self.accuracy_ordering[map_cur_to_new_ordering]
            self.predictions = self.predictions[:, map_cur_to_new_ordering]
        
    def tuned_refit(self, X_to_add, y_to_add):
        # online update of accuracies, 
        # then changing the corresponding ordering 
        # there *must* be cool algorithms for this that have already been studied
        predictions_to_add = np.zeros((X_to_add.shape[0], self.predictions.shape[1]))
        self.y = pd.concat([self.y, y_to_add], ignore_index=True)
        self.X = pd.concat([self.X, X_to_add], ignore_index=True)

        for j in range(len(self.all_trees)): 
            predictions_to_add[:, j] = _predict(self.all_trees[self.accuracy_ordering[j]], X_to_add).values
        self.predictions = np.concatenate([self.predictions, predictions_to_add])

        # get accuracies for matrix and resort matrix/accuracies accordingly
        accuracies = (self.predictions == self.y.values[:, np.newaxis]).mean(axis=0) #accuracies for current accuracy ordering
        map_cur_to_new_ordering = accuracies.argsort()
        self.accuracy_ordering = self.accuracy_ordering[map_cur_to_new_ordering]
        self.predictions = self.predictions[:, map_cur_to_new_ordering]
        self._tune_eps(accuracies[map_cur_to_new_ordering])

