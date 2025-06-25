from lfr.treefarms_wrapper import Treefarms_LFR
import pandas as pd

X = pd.DataFrame([
    [1, 0],
    [0, 0],
    [0, 1],
])
y = pd.Series([
    1,
    1,
    0
])

def test_predict(): 
    model = Treefarms_LFR()
    model.fit(X, y, 0.01)
    predictions = model.all_predictions_one_sample(pd.Series([0, 1]))
    assert(predictions[0] == 0)

def test_predict_matrix(): 
    model = Treefarms_LFR()
    model.fit(X, y, 0.01)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())

def test_refit_larger_eps(): 
    model = Treefarms_LFR()
    model.fit(X, y, 0.01)
    model.refit(X, y, 0.02)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())

def test_refit_smaller_eps():
    model = Treefarms_LFR()
    model.fit(X, y, 0.01)
    model.refit(X, y, 0)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())