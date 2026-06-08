import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error


def get_cv_rmse(model_object, X_train, y_train, k=5):
    """
    Calculates the K-fold CV RMSE for the current training set.

    Args:
        model_object: The scikit-learn compatible model object
                      (e.g., the .model inside your RidgeRegressionPredictor).
        X_train (pd.DataFrame): Current training features.
        y_train (pd.Series): Current training labels.
        k (int): Number of folds.
    """

    ### Set up ###
    if len(y_train) < k * 2:
        return np.nan
    model_to_cv = model_object

    ### Define the scoring ###
    scores = cross_val_score(
        model_to_cv,
        X_train,
        y_train,
        cv=KFold(n_splits=k, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )
    return -np.mean(scores)


import torch
import torch.nn as nn
import torch.optim as optim


def get_cv_rmse_NN(model_object, X_train, y_train, k=5, epochs=50, batch_size=32, lr=0.001):
    """
    Calculates the K-fold CV RMSE for a PyTorch model.

    Args:
        model_object: The PyTorch model (e.g., MLPredictor instance).
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        k (int): Number of folds.
        epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.

    Returns:
        float: Mean RMSE across all folds.
    """
    if len(y_train) < k * 2:
        return np.nan

    # Convert to numpy if DataFrame/Series
    if hasattr(X_train, "values"):
        X_train = X_train.values.astype(np.float32)
    if hasattr(y_train, "values"):
        y_train = y_train.values.astype(np.float32).reshape(-1, 1)

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in kf.split(X_train):
        # Split data
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(device)

        # Create a fresh model instance for each fold
        model = model_object.__class__(
            input_size=model_object.input_size,
            hidden_size=model_object.hidden_size,
            output_size=model_object.output_size,
            lr=lr,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            mse = criterion(val_preds, y_val_tensor)
            rmse = torch.sqrt(mse).item()
            rmse_scores.append(rmse)

    return np.mean(rmse_scores)
