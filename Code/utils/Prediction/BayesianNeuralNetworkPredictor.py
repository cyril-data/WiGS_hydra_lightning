### Libraries ###
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd 

### Model Definition ###
class _BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=50, dropout_rate=0.2):
        super(_BayesianNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=1) # Softmax is applied in log_softmax later

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        return x 

class BayesianNeuralNetworkPredictor:
    def __init__(self, Seed: int, hidden_size: int = 50, dropout_rate: float = 0.2, 
                 epochs: int = 100, learning_rate: float = 0.001, batch_size_train: int = 32):
        
        # Store configuration parameters
        self.Seed = Seed
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size_train = batch_size_train
        
        # Initialize model components
        self.model = None
        self.input_size = None
        self.num_classes = None

        # Set seed 
        torch.manual_seed(self.Seed)
        np.random.seed(self.Seed) 

    def fit(self, X_train_df: pd.DataFrame, y_train_series: pd.Series):
        
        # Extract X and Y
        X_train_np = X_train_df.values
        y_train_np = y_train_series.values

        # Determine input size and number of classes
        self.input_size = X_train_np.shape[1]
        self.num_classes = len(np.unique(y_train_np))

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.long) # Ensure y is long for CrossEntropyLoss

        # Initialize the BNN model 
        if self.model is None or \
           self.model.fc1.in_features != self.input_size or \
           self.model.fc2.out_features != self.num_classes:
            self.model = _BayesianNeuralNetwork(
                input_size=self.input_size,
                num_classes=self.num_classes,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate
            )

        # Training Function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Set model to training mode (dropout is active)
        self.model.train() 
        for epoch in range(self.epochs):

            # Shuffle and batch data
            permutation = torch.randperm(X_train_tensor.size(0))
            for i in range(0, X_train_tensor.size(0), self.batch_size_train):
                indices = permutation[i:i+self.batch_size_train]
                batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X_data_df: pd.DataFrame) -> np.ndarray:
        X_data_np = X_data_df.values
        log_probs_N_K_C = self.predict_proba_K(X_data_np, K_samples=100) # Use a default K for point prediction
        probs_N_K_C = torch.exp(log_probs_N_K_C)
        mean_probs_N_C = torch.mean(probs_N_K_C, dim=1)
        return torch.argmax(mean_probs_N_C, dim=1).cpu().numpy()

    def predict_proba(self, X_data_df: pd.DataFrame) -> np.ndarray:
        X_data_np = X_data_df.values
        log_probs_N_K_C = self.predict_proba_K(X_data_np, K_samples=100) # Use a default K for point prediction
        probs_N_K_C = torch.exp(log_probs_N_K_C)
        mean_probs_N_C = torch.mean(probs_N_K_C, dim=1)
        return mean_probs_N_C.cpu().numpy()

    def predict_proba_K(self, X_data_np: np.ndarray, K_samples: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Set model to training mode to enable dropout for MC Dropout
        self.model.train() 
        X_data_tensor = torch.tensor(X_data_np, dtype=torch.float32)

        N_data = X_data_tensor.shape[0]
        num_classes = self.model.fc2.out_features
        log_probs_N_K_C = torch.empty((N_data, K_samples, num_classes), dtype=torch.float32) # Changed to float32 for consistency

        with torch.no_grad():
            for k in range(K_samples):
                logits = self.model(X_data_tensor)
                log_softmax_output_k = torch.log_softmax(logits, dim=1)
                log_probs_N_K_C[:, k, :] = log_softmax_output_k
        
        return log_probs_N_K_C