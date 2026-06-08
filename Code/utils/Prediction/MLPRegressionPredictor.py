import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


class MLPredictor(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) model using PyTorch.
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 1, **kwargs):
        super(MLPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Optional dropout for regularization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLPRegressionPredictor:
    """
    A wrapper for the PyTorch MLP model.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        output_size: int = 1,
        lr: float = 0.001,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.model = MLPredictor(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(
        self,
        X_train_df: pd.DataFrame,
        y_train_series: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        X_train = X_train_df.values.astype(np.float32)
        y_train = y_train_series.values.astype(np.float32).reshape(-1, 1)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_data = X_data_df.values.astype(np.float32)
        X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_data_tensor)

        return predictions.cpu().numpy().flatten()
