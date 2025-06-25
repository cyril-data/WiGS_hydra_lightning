from lfr.tuning_treefarms_wrapper import tuning_Treefarms_LFR
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
    model = tuning_Treefarms_LFR(static_config={'verbose': True})
    model.fit(X, y, 0.01)
    predictions = model.all_predictions_one_sample(pd.Series([0, 1]))
    assert(predictions[0] == 0)

def test_predict_matrix(): 
    model = tuning_Treefarms_LFR(static_config={'verbose': True})
    model.fit(X, y, 0.01)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())

def test_refit_larger_eps(): 
    model = tuning_Treefarms_LFR(static_config={'verbose': True})
    model.fit(X, y, 0.01)
    model.refit(X, y, 0.02)
    model.tuned_refit(X, y)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())

def test_refit_smaller_eps():
    model = tuning_Treefarms_LFR({'verbose': True})
    model.fit(X, y, 0.01)
    model.refit(X, y, 0)
    model.tuned_refit(X, y)
    predictions = model.all_predictions(X)
    assert((predictions.iloc[:, 0] == y).all())