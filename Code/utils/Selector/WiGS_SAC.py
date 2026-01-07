### Import Libraries ###
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.spatial.distance import cdist
from utils.Auxiliary.DataFrameUtils import get_features_and_target
from collections import deque
import random

# --- Hyperparameters ---
HIDDEN_SIZE = 64          # Number of neurons in hidden layers
BUFFER_SIZE = 10000       # Max size of the replay buffer
BATCH_SIZE = 64           # Number of samples to train on from the buffer
LEARNING_RATE = 3e-4      # Learning rate for actor and critic networks
GAMMA = 0.99              # Discount factor for future rewards
TAU = 0.005               # Target network soft update rate
ALPHA = 0.2               # Entropy regularization coefficient (the "temperature")

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Actor and Critic Network ###
class Actor(nn.Module):
    """
    The Actor (Policy) network. It maps a state to an action.
    It outputs the parameters of a distribution from which the action is sampled.
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)   
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    """
    The Critic (Q-Value) network. It maps a (state, action) pair to a Q-value.
    SAC uses a "twin critic" setup, so we define one class and instantiate it twice.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        # Critic 2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

### Replay Buffer ###

class ReplayBuffer:
    """A simple replay buffer to store experience tuples."""
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

### Main WiGS SAC Selector Class ###

class WiGS_SAC_Selector:
    """
    Implements a WiGS selector using a Soft Actor-Critic (SAC) agent for weight selection. 🤖
    """
    def __init__(self, initial_candidate_size: int, Seed: int = None, **kwargs):
        """
        Initializes the WiGS_SAC_Selector.
        Args:
            initial_candidate_size (int): The total number of candidates at the start.
            Seed (int, optional): A random seed for reproducibility.
            **kwargs: Accepts and ignores additional keyword arguments for consistency.
        """
        if Seed is not None:
            torch.manual_seed(Seed)
            np.random.seed(Seed)
            random.seed(Seed)

        # We will determine state_dim later, once we see the first dataframe
        self.state_dim = None
        self.action_dim = 1  # w_x is a single continuous value

        # SAC components
        self.actor = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # State tracking for the active learning loop
        self.initial_candidate_size = initial_candidate_size
        self.iteration = 0
        self.last_state = None
        self.last_action = None
        self.last_rmse = None

    def _initialize_agent(self, state_dim: int):
        """Initializes all networks and optimizers once the state dimension is known."""
        print(f"SAC Agent Initializing with State Dimension: {state_dim}")
        self.state_dim = state_dim
        self.actor = Actor(state_dim, self.action_dim).to(DEVICE)
        self.critic = Critic(state_dim, self.action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, self.action_dim).to(DEVICE)

        # Initialize target networks to be identical to main networks
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def _get_state(self, df_Train: pd.DataFrame, df_Candidate: pd.DataFrame, current_rmse: float) -> np.ndarray:
        """
        Constructs the state vector from the current AL environment.
        This is a critical part of the design and can be expanded.
        """
        X_train, y_train = get_features_and_target(df=df_Train, target_column_name="Y")

        # --- State Features ---
        # 1. Current model performance
        state_rmse = np.array([current_rmse])
        # 2. AL Process Progress
        progress = np.array([self.iteration / self.initial_candidate_size])
        # 3. Labeled set statistics (captures data distribution)
        labeled_features_mean = X_train.mean().values
        labeled_features_std = X_train.std().values
        labeled_target_mean = np.array([y_train.mean()])
        labeled_target_std = np.array([y_train.std()])
        
        # Replace NaNs that might occur in std dev at the start
        labeled_features_std = np.nan_to_num(labeled_features_std, nan=0.0)
        labeled_target_std = np.nan_to_num(labeled_target_std, nan=0.0)

        # Concatenate all features into a single state vector
        state = np.concatenate([
            state_rmse,
            progress,
            labeled_features_mean,
            labeled_features_std,
            labeled_target_mean,
            labeled_target_std
        ]).flatten()
        
        return state

    def update(self):
        """Samples a batch from the replay buffer and updates the agent's networks."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return  
        
        # Sample a batch
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        
        # Convert to PyTorch tensors
        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        action = action.reshape(-1, self.action_dim) # <-- ADD THIS LINE TO FIX SHAPE
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)
        # --- Update Critic ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next_min = torch.min(q1_next, q2_next)
            # SAC target: r + gamma * (1-d) * (Q_next - alpha * log_prob)
            target_q = reward + (1 - done) * GAMMA * (q_next_min - ALPHA * next_log_prob)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi_min = torch.min(q1_pi, q2_pi)
        # Actor loss: alpha * log_prob - Q_value
        actor_loss = ((ALPHA * log_pi) - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ## Soft Update Target Networks ##
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


    def select(self, df_Candidate: pd.DataFrame, Model=None, df_Train: pd.DataFrame = None, current_rmse: float = None) -> dict:
        """
        Selects a point by first choosing a weight `w_x` via the SAC agent.
        """
        ### If there are no more observations in the candidate set ###
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Construct current state and initialize agent on first run ##
        current_state = self._get_state(df_Train, df_Candidate, current_rmse)
        if self.actor is None:
            self._initialize_agent(len(current_state))
        
        ## Store experience from the PREVIOUS step and update agent ##
        if self.last_state is not None:
            reward = self.last_rmse - current_rmse  # Reward is the decrease in RMSE
            done = False  # An episode is a full AL run, so 'done' is always false here.
            
            self.replay_buffer.push(self.last_state, self.last_action, reward, current_state, done)
            self.update() # Perform one learning step

        ## Select an action (w_x) for the CURRENT step ##
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor)
        
        # Action is in [-1, 1], scale to [0, 1] for w_x
        w_x_tensor = (action.cpu().numpy().flatten()[0] + 1) / 2
        w_x = np.clip(w_x_tensor, 0, 1) # Ensure w_x is in [0, 1]
        w_y = 1.0 - w_x
        
        ## Store state and action for the next iteration's update ##
        self.last_state = current_state
        self.last_action = action.cpu().numpy() 
        self.last_rmse = current_rmse
        
        ### WiGS Point Selection Logic ###

        X_Candidate, _ = get_features_and_target(df=df_Candidate, target_column_name="Y")
        X_Train, y_Train = get_features_and_target(df=df_Train, target_column_name="Y")
    
        d_nmX = cdist(X_Candidate.values, X_Train.values, metric='euclidean')
        Predictions = Model.predict(X_Candidate)
        d_nmY = cdist(Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric='euclidean')

        epsilon = 1e-8
        d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        final_scores = score_matrix.min(axis=1)
        best_candidate_iloc = np.argmax(final_scores)

        self.iteration += 1

        IndexRecommendation = df_Candidate.iloc[[best_candidate_iloc]].index[0]
        return {"IndexRecommendation": [float(IndexRecommendation)], "w_x": w_x}