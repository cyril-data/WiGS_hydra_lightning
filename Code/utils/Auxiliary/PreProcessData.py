### Import Packages ###
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

### Find optimal splits ###
def FindOptimalSplits(X, y, NTrees=40, MaxDepth=1):
    
    ### Set Up ###
    SplitDictionary = {}
    
    ### Gradient Boosting ###
    GradientBoostingModel = GradientBoostingClassifier(n_estimators=NTrees, 
                                   max_depth=MaxDepth,
                                   random_state=0)
    GradientBoostingModel.fit(X, y)
    
    ### Extract Splits ###
    for FeatureIndex in range(X.shape[1]):
        SplitsSet = set()
        
        ## Collect split points from each tree ##
        for tree in GradientBoostingModel.estimators_.flatten():
            tree = tree.tree_
            
            # Check each node in the tree #
            for NodeId in range(tree.node_count):
                # If it's a split node and splits on our feature of interest
                if tree.feature[NodeId] == FeatureIndex:
                    # Round the threshold to nearest integer
                    SplitsSet.add(round(tree.threshold[NodeId]))
        
        SplitDictionary[X.columns[FeatureIndex]] = sorted(list(SplitsSet))
    
    return SplitDictionary

### Create binary features ###
def CreateBinaryFeatures(X, SplitDictionary):
    
    ### Set Up ###
    BinaryFeatures = pd.DataFrame()
    
    ### Process Each Feature ###
    for Column, SplitsSet in SplitDictionary.items():
        for Split in SplitsSet:
            ## Create feature name and values ##
            FeatureName = f"{Column}_leq_{Split}"
            BinaryFeatures[FeatureName] = (X[Column] <= Split).astype(int)
    
    return BinaryFeatures

### Preprocess the data ###
def PreProcessData(df):

    ### Separate features and target ###
    X = df.drop('Y', axis=1)
    y = df['Y']
    
    ### Find optimal splits using gradient boosting ###
    SplitDictionary = FindOptimalSplits(X, y)
    
    ### Create binary features ###
    BinaryFeatures = CreateBinaryFeatures(X, SplitDictionary)
    
    ### Add target variable back ###
    BinaryFeatures['Y'] = y
    
    return BinaryFeatures