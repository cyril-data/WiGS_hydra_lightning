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
import faiss
from utils.Prediction.LightHydra import (
    hl_pd_to_dataloader,
    hl_y_pred_pd_to_tensor,
    hl_np_to_dataloader,
)
import time

# --- Hyperparameters ---
HIDDEN_SIZE = 64  # Number of neurons in hidden layers
BUFFER_SIZE = 10000  # Max size of the replay buffer
BATCH_SIZE = 64  # Number of samples to train on from the buffer
LEARNING_RATE = 3e-4  # Learning rate for actor and critic networks
GAMMA = 0.99  # Discount factor for future rewards
TAU = 0.005  # Target network soft update rate
ALPHA = 0.2  # Entropy regularization coefficient (the "temperature")

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_min_max_dist_gpu(candidates_np, ref_np, batch_size, train_chunk_size=None, label=""):
    if train_chunk_size is None:
        train_chunk_size = batch_size

    ref_t = torch.from_numpy(ref_np).to(DEVICE).half()
    ref_sq = (ref_t**2).sum(dim=1)
    n_ref = ref_t.shape[0]

    g_min = torch.tensor(float("inf"), device=DEVICE)
    g_max = torch.tensor(float("-inf"), device=DEVICE)

    with torch.no_grad():
        for i in range(0, len(candidates_np), batch_size):
            # print(f"global_min_max_dist_gpu : {i} sur {int(len(candidates_np)/ batch_size)}")
            b = (
                torch.from_numpy(candidates_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .half()
            )
            b_sq = (b**2).sum(dim=1, keepdim=True)

            for j in range(0, n_ref, train_chunk_size):
                ref_chunk = ref_t[j : j + train_chunk_size]
                ref_sq_chunk = ref_sq[j : j + train_chunk_size]

                dist_sq = b_sq + ref_sq_chunk.unsqueeze(0) - 2.0 * (b @ ref_chunk.T)
                dist_sq.clamp_(min=0)
                dist = dist_sq.sqrt_()

                g_min = torch.minimum(g_min, dist.min())
                g_max = torch.maximum(g_max, dist.max())

                del dist_sq, dist

            del b, b_sq

    return g_min.item(), g_max.item()


def compute_final_scores_gpu(
    X_candidate_np,
    X_train_np,
    dX_min,
    dX_max,
    Y_candidate_np,
    Y_train_np,
    dY_min,
    dY_max,
    w_x,
    w_y,
    batch_size,
    train_chunk_size=None,
    epsilon=1e-8,
):
    if train_chunk_size is None:
        train_chunk_size = batch_size

    X_train_t = torch.from_numpy(X_train_np).to(DEVICE).half()
    X_train_sq = (X_train_t**2).sum(dim=1)
    n_train_x = X_train_t.shape[0]

    Y_train_t = torch.from_numpy(Y_train_np).to(DEVICE).half()
    Y_train_sq = (Y_train_t**2).sum(dim=1)
    n_train_y = Y_train_t.shape[0]

    assert (
        n_train_x == n_train_y
    ), "X_train et Y_train doivent avoir le même nombre de lignes (même index)"

    dX_range = dX_min and (dX_max - dX_min + epsilon)
    dX_range = dX_max - dX_min + epsilon
    dY_range = dY_max - dY_min + epsilon

    n = len(X_candidate_np)
    final_scores = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            bX = (
                torch.from_numpy(X_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .half()
            )
            bX_sq = (bX**2).sum(dim=1, keepdim=True)

            bY = (
                torch.from_numpy(Y_candidate_np[i : i + batch_size])
                .to(DEVICE, non_blocking=True)
                .half()
            )
            bY_sq = (bY**2).sum(dim=1, keepdim=True)

            running_min = torch.full(
                (bX.shape[0],), float("inf"), device=DEVICE, dtype=torch.float16
            )

            for j in range(0, n_train_x, train_chunk_size):
                Xt_chunk = X_train_t[j : j + train_chunk_size]
                Xsq_chunk = X_train_sq[j : j + train_chunk_size]
                Yt_chunk = Y_train_t[j : j + train_chunk_size]
                Ysq_chunk = Y_train_sq[j : j + train_chunk_size]

                dX_chunk = (
                    (bX_sq + Xsq_chunk.unsqueeze(0) - 2.0 * (bX @ Xt_chunk.T))
                    .clamp_(min=0)
                    .sqrt_()
                )
                dY_chunk = (
                    (bY_sq + Ysq_chunk.unsqueeze(0) - 2.0 * (bY @ Yt_chunk.T))
                    .clamp_(min=0)
                    .sqrt_()
                )

                score_chunk = (dX_chunk - dX_min) * (w_x / dX_range) + (dY_chunk - dY_min) * (
                    w_y / dY_range
                )

                running_min = torch.minimum(running_min, score_chunk.min(dim=1).values)

                del dX_chunk, dY_chunk, score_chunk

            final_scores[i : i + batch_size] = running_min.float().cpu().numpy()

            del bX, bY, bX_sq, bY_sq, running_min

    return final_scores


# def global_min_max_dist_gpu(candidates_np, ref_np, batch_size, label=""):
#     ref_t = torch.from_numpy(ref_np).to(DEVICE).half()
#     ref_sq = (ref_t**2).sum(dim=1)  # précalculé une seule fois

#     g_min = torch.tensor(float("inf"), device=DEVICE)
#     g_max = torch.tensor(float("-inf"), device=DEVICE)

#     with torch.no_grad():
#         for i in range(0, len(candidates_np), batch_size):
#             b = (
#                 torch.from_numpy(candidates_np[i : i + batch_size])
#                 .to(DEVICE, non_blocking=True)
#                 .half()
#             )
#             b_sq = (b**2).sum(dim=1, keepdim=True)

#             dist_sq = b_sq + ref_sq.unsqueeze(0) - 2.0 * (b @ ref_t.T)
#             dist_sq.clamp_(min=0)
#             dist = dist_sq.sqrt_()

#             g_min = torch.minimum(g_min, dist.min())
#             g_max = torch.maximum(g_max, dist.max())

#             del b, b_sq, dist_sq, dist

#     return g_min.item(), g_max.item()


# def compute_final_scores_gpu(
#     X_candidate_np,
#     X_train_np,
#     dX_min,
#     dX_max,
#     Y_candidate_np,
#     Y_train_np,
#     dY_min,
#     dY_max,
#     w_x,
#     w_y,
#     batch_size,
#     epsilon=1e-8,
# ):
#     X_train_t = torch.from_numpy(X_train_np).to(DEVICE).half()
#     X_train_sq = (X_train_t**2).sum(dim=1)

#     Y_train_t = torch.from_numpy(Y_train_np).to(DEVICE).half()
#     Y_train_sq = (Y_train_t**2).sum(dim=1)

#     dX_range = dX_max - dX_min + epsilon
#     dY_range = dY_max - dY_min + epsilon

#     n = len(X_candidate_np)
#     final_scores = np.empty(n, dtype=np.float32)

#     with torch.no_grad():
#         for i in range(0, n, batch_size):
#             bX = (
#                 torch.from_numpy(X_candidate_np[i : i + batch_size])
#                 .to(DEVICE, non_blocking=True)
#                 .half()
#             )
#             bX_sq = (bX**2).sum(dim=1, keepdim=True)
#             dX_batch = (
#                 (bX_sq + X_train_sq.unsqueeze(0) - 2.0 * (bX @ X_train_t.T)).clamp_(min=0).sqrt_()
#             )

#             bY = (
#                 torch.from_numpy(Y_candidate_np[i : i + batch_size])
#                 .to(DEVICE, non_blocking=True)
#                 .half()
#             )
#             bY_sq = (bY**2).sum(dim=1, keepdim=True)
#             dY_batch = (
#                 (bY_sq + Y_train_sq.unsqueeze(0) - 2.0 * (bY @ Y_train_t.T)).clamp_(min=0).sqrt_()
#             )

#             # normalisation + pondération fusionnées (pas de tenseur intermédiaire séparé)
#             score_batch = (dX_batch - dX_min) * (w_x / dX_range) + (dY_batch - dY_min) * (
#                 w_y / dY_range
#             )

#             final_scores[i : i + batch_size] = score_batch.min(dim=1).values.cpu().numpy()

#             del bX, bY, dX_batch, dY_batch, score_batch

#     return final_scores


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
            nn.ReLU(),
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
            nn.Linear(HIDDEN_SIZE, 1),
        )
        # Critic 2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
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
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


### Main WiGS SAC Selector Class ###
class WiGS_SAC_Selector:
    """
    Implements a WiGS selector using a Soft Actor-Critic (SAC) agent for weight selection.
    """

    def __init__(
        self, initial_candidate_size: int, Seed: int = None, k_top_candidate=10, **kwargs
    ):
        """+
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

        self.state_dim = None
        self.action_dim = 1

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

        self.k_top_candidate = k_top_candidate

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

    def _get_state(
        self,
        df_Train: pd.DataFrame,
        df_Candidate: pd.DataFrame,
        current_rmse: float,
        y_size: int = 1,
    ) -> np.ndarray:
        """
        Constructs the state vector from the current AL environment.
        This is a critical part of the design and can be expanded.
        """
        # X_train, y_train = get_features_and_target(df_Train, "y_size")
        X_train, y_train = get_features_and_target(df_Train, y_size=y_size)

        # 1. Current model performance
        state_rmse = np.array([current_rmse])
        # 2. AL Process Progress
        progress = np.array([self.iteration / self.initial_candidate_size])
        # 3. Labeled set statistics (captures data distribution)
        labeled_features_mean = X_train.mean().values
        labeled_features_std = X_train.std().values
        labeled_target_mean = np.array([y_train.mean()]).flatten()
        labeled_target_std = np.array([y_train.std()]).flatten()
        labeled_features_std = np.nan_to_num(labeled_features_std, nan=0.0)
        labeled_target_std = np.nan_to_num(labeled_target_std, nan=0.0)

        # TODO WARNING .flatten() IS NOT GOOD, check the good scalar for multiouput target

        # Concatenate all features into a single state vector
        state = np.concatenate(
            [
                state_rmse,
                progress,
                labeled_features_mean,
                labeled_features_std,
                labeled_target_mean,
                labeled_target_std,
            ]
        ).flatten()

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
        action = action.reshape(-1, self.action_dim)
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

    def select(
        self,
        df_Candidate: pd.DataFrame,
        y_size: int,
        Model=None,
        df_Train: pd.DataFrame = None,
        current_rmse: float = None,
        SimulationConfigInputUpdated: dict = None,
    ) -> dict:
        """
        Selects a point by first choosing a weight `w_x` via the SAC agent.
        """

        ## 1. Initialization
        StartTime = time.time()

        hl_data = None

        ### If there are no more observations in the candidate set ###
        if df_Candidate.empty:
            return {"IndexRecommendation": []}

        ## Construct current state and initialize agent on first run ##
        current_state = self._get_state(df_Train, df_Candidate, current_rmse)
        if self.actor is None:
            self._initialize_agent(len(current_state))

        ## Store experience from the PREVIOUS step and update agent ##
        if self.last_state is not None:
            reward = self.last_rmse - current_rmse
            done = False

            self.replay_buffer.push(self.last_state, self.last_action, reward, current_state, done)
            self.update()

        ## Select an action (w_x) for the CURRENT step ##
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor)

        # Action is in [-1, 1], scale to [0, 1] for w_x
        w_x_tensor = (action.cpu().numpy().flatten()[0] + 1) / 2
        w_x = np.clip(w_x_tensor, 0, 1)
        w_y = 1.0 - w_x

        ## Store state and action for the next iteration's update ##
        self.last_state = current_state
        self.last_action = action.cpu().numpy()
        self.last_rmse = current_rmse

        ### ── VERSION BATCH (memory-efficient) ──────────────────────────────────────
        X_Candidate, _ = get_features_and_target(df_Candidate, y_size=None)
        X_Train, y_Train = get_features_and_target(df_Train, y_size=y_size)

        X_Candidate_f32 = X_Candidate.values.astype(np.float32)
        X_Train_f32 = X_Train.values.astype(np.float32)

        select_ytrain_cols = None

        print(f"\t+++ Wigs_SAC #1 : {time.time() - StartTime} +++")

        ## 2. Prediction on candidates
        StartTime = time.time()

        if SimulationConfigInputUpdated["hl_trainer"] is not None:

            hl_data = SimulationConfigInputUpdated["hl_data"]

            # Après (réutilise le même DataLoader)
            candidate_indices = X_Candidate.index.tolist()
            hl_data.pred_data.update_indices(candidate_indices)

            y_pred = Model.predict(model=Model.model, dataloaders=hl_data)

            y_pred, select_ytrain_cols = hl_y_pred_pd_to_tensor(
                y_pred, y_Train.columns.to_list(), X_Candidate.index
            )
            # restrain only on regression
            Predictions = y_pred[select_ytrain_cols].values

        else:
            Predictions = Model.predict(X_Candidate)
        print(f"\t+++ Wigs_SAC Prediction on candidates : {time.time() - StartTime} +++")

        ## 3. Distance and score calculation
        StartTime = time.time()
        pred_vals = (
            Predictions.reshape(-1, 1).astype(np.float32)
            if len(Predictions.shape) == 1
            else Predictions.astype(np.float32)
        )
        if select_ytrain_cols is not None:
            y_Train = y_Train[select_ytrain_cols]
        y_train_values = (
            y_Train.values.reshape(-1, 1).astype(np.float32)
            if len(y_Train.shape) == 1
            else y_Train.values.astype(np.float32)
        )

        epsilon = 1e-8

        batch_size = 512

        print("batch_size DIST ! ", batch_size)

        # Pass 1 : min/max global
        dX_global_min, dX_global_max = np.inf, -np.inf
        dY_global_min, dY_global_max = np.inf, -np.inf

        dX_global_min, dX_global_max = global_min_max_dist_gpu(
            X_Candidate_f32, X_Train_f32, batch_size
        )
        print("dX_global_min", dX_global_min)

        print(f"\t+++ Wigs_SAC dX_global_min : {time.time() - StartTime} +++")

        StartTime = time.time()

        dY_global_min, dY_global_max = global_min_max_dist_gpu(
            pred_vals, y_train_values, batch_size
        )
        print(f"\t+++ Wigs_SAC dY_global_min : {time.time() - StartTime} +++")

        print("dY_global_min", dY_global_min)

        # print("torch dX_global_min", dX_global_min)
        # print("torch dX_global_max", dX_global_max)

        # print("torch dY_global_min", dY_global_min)
        # print("torch dY_global_max", dY_global_max)

        # for i in range(0, len(X_Candidate_f32), batch_size):
        #     StartTime1 = time.time()
        #     bX = X_Candidate_f32[i : i + batch_size]
        #     dX_batch = cdist(bX, X_Train_f32, metric="euclidean")
        #     dX_global_min = min(dX_global_min, dX_batch.min())
        #     dX_global_max = max(dX_global_max, dX_batch.max())

        #     bY = pred_vals[i : i + batch_size]
        #     dY_batch = cdist(bY, y_train_values, metric="euclidean")
        #     dY_global_min = min(dY_global_min, dY_batch.min())
        #     dY_global_max = max(dY_global_max, dY_batch.max())
        #     print(f"\t+++ Wigs_SAC dY_global_max batch {i} : {time.time() - StartTime1} +++")

        # print("numpy dX_global_min", dX_global_min)
        # print("numpy dX_global_max", dX_global_max)

        # print("numpy dY_global_min", dY_global_min)
        # print("numpy dY_global_max", dY_global_max)

        print(f"\t+++ Wigs_SAC Distance min calculation : {time.time() - StartTime} +++")

        ## 4. Distance and score calculation
        StartTime = time.time()

        # Pass 2 : scores
        final_scores_batch = np.empty(len(X_Candidate_f32), dtype=np.float32)

        final_scores_batch = compute_final_scores_gpu(
            X_Candidate_f32,
            X_Train_f32,
            dX_global_min,
            dX_global_max,
            pred_vals,
            y_train_values,
            dY_global_min,
            dY_global_max,
            w_x,
            w_y,
            batch_size=batch_size,
        )

        print("final_scores_batch", final_scores_batch)

        # print("torch final_scores_batch", final_scores_batch)
        # print("torch dX_global_max", dX_global_max)

        # print("torch dY_global_min", dY_global_min)
        # print("torch dY_global_max", dY_global_max)

        # for i in range(0, len(X_Candidate_f32), batch_size):
        #     bX = X_Candidate_f32[i : i + batch_size]
        #     dX_batch = cdist(bX, X_Train_f32, metric="euclidean")
        #     d_prime_X = (dX_batch - dX_global_min) / (dX_global_max - dX_global_min + epsilon)

        #     bY = pred_vals[i : i + batch_size]
        #     dY_batch = cdist(bY, y_train_values, metric="euclidean")
        #     d_prime_Y = (dY_batch - dY_global_min) / (dY_global_max - dY_global_min + epsilon)

        #     score_batch = (w_x * d_prime_X) + (w_y * d_prime_Y)
        #     final_scores_batch[i : i + batch_size] = score_batch.min(axis=1)

        # print("numpy final_scores_batch", final_scores_batch)

        top_k_number_batch = self.k_top_candidate
        if len(final_scores_batch) < self.k_top_candidate:
            top_k_number_batch = len(final_scores_batch)

        print(f"\t+++ Wigs_SAC final_scores_batch : {time.time() - StartTime} +++")
        # best_candidate_iloc_batch = np.argpartition(final_scores_batch, top_k_number_batch)[
        #     :top_k_number_batch
        # ]
        best_candidate_iloc_batch = np.argpartition(final_scores_batch, -top_k_number_batch)[
            -top_k_number_batch:
        ]

        # # top_k_indices_batch = np.argsort(final_scores_batch)[:top_k_number_batch]
        # top_k_indices_batch = np.argsort(final_scores_batch)[-top_k_number_batch:][::-1]
        # top_k_scores_batch = final_scores_batch[top_k_indices_batch]

        # print("═" * 60)
        # print(f"[BATCH] Top-{top_k_number_batch} candidats sélectionnés :")
        # print("best_candidate_iloc_batch", best_candidate_iloc_batch)
        # for rank, (idx, score) in enumerate(zip(top_k_indices_batch, top_k_scores_batch)):
        #     print(
        #         f"  #{rank+1:3d} | iloc={idx:5d} | score={score:.8f} | index={df_Candidate.iloc[idx].name}"
        #     )
        # print("═" * 60)

        # ### ── ARGMAX ────────────────────────────────────────────────────────────

        # d_nmX = cdist(X_Candidate.values, X_Train.values, metric="euclidean")
        # d_nmY = cdist(
        #     Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric="euclidean"
        # )

        # epsilon = 1e-8
        # d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        # d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        # score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        # final_scores = score_matrix.min(axis=1)
        # best_candidate_iloc = np.argmax(final_scores)

        # best_candidate_iloc_batch = [best_candidate_iloc]

        # print("═" * 60)
        # print(f"[MAX] Top-{top_k_number_batch} candidats sélectionnés :")
        # print(
        #     f" iloc={best_candidate_iloc:5d} | score={final_scores[best_candidate_iloc]}  | df_Candidate {df_Candidate.iloc[[best_candidate_iloc]].index[0]}"
        # )
        # # print(
        # #     f"  #{rank+1:3d} | iloc={best_candidate_iloc:5d} | score={final_scores:.8f} | index={df_Candidate.iloc[best_candidate_iloc].name}"
        # # )
        # print("═" * 60)

        # ### ── VERSION CDIST FULL ─────────────────────────────────────────────────────
        # X_Candidate, _ = get_features_and_target(df_Candidate, y_size=None)
        # X_Train, y_Train = get_features_and_target(df_Train, y_size=y_size)

        # d_nmX = cdist(X_Candidate.values, X_Train.values, metric="euclidean")

        # Predictions = Model.predict(X_Candidate)
        # d_nmY = cdist(
        #     Predictions.reshape(-1, 1), y_Train.values.reshape(-1, 1), metric="euclidean"
        # )

        # epsilon = 1e-8
        # d_prime_nmX = (d_nmX - d_nmX.min()) / (d_nmX.max() - d_nmX.min() + epsilon)
        # d_prime_nmY = (d_nmY - d_nmY.min()) / (d_nmY.max() - d_nmY.min() + epsilon)

        # score_matrix = (w_x * d_prime_nmX) + (w_y * d_prime_nmY)
        # final_scores_cdist = score_matrix.min(axis=1)

        # top_k_number_cdist = self.k_top_candidate
        # if len(final_scores_cdist) < self.k_top_candidate:
        #     top_k_number_cdist = len(final_scores_cdist)

        # # best_candidate_iloc_batch = np.argpartition(final_scores_batch, top_k_number_batch)[
        # #     :top_k_number_batch
        # # ]
        # best_candidate_iloc_batch = np.argpartition(final_scores_batch, -top_k_number_batch)[
        #     -top_k_number_batch:
        # ]

        # # top_k_indices_batch = np.argsort(final_scores_batch)[:top_k_number_batch]
        # top_k_indices_batch = np.argsort(final_scores_batch)[-top_k_number_batch:][::-1]
        # top_k_scores_batch = final_scores_batch[top_k_indices_batch]

        # # best_candidate_iloc_cdist = np.argpartition(final_scores_cdist, -top_k_number_cdist)

        # # top_k_indices_cdist = np.argsort(final_scores_cdist)[:top_k_number_cdist]
        # # top_k_scores_cdist = final_scores_cdist[top_k_indices_cdist]

        # print("═" * 60)
        # print(f"[CDIST] Top-{top_k_number_cdist} candidats sélectionnés :")
        # for rank, (idx, score) in enumerate(zip(top_k_indices_cdist, top_k_scores_cdist)):
        #     print(
        #         f"  #{rank+1:3d} | iloc={idx:5d} | score={score:.8f} | index={df_Candidate.iloc[idx].name}"
        #     )
        # print("═" * 60)

        # ### ── COMPARAISON ────────────────────────────────────────────────────────────
        # print(
        #     "\n[DIFF] Écart max entre les scores :",
        #     np.abs(final_scores_batch - final_scores_cdist).max(),
        # )
        # print(
        #     "[DIFF] Top-k identiques ?",
        #     set(top_k_indices_batch.tolist()) == set(top_k_indices_cdist.tolist()),
        # )

        ## ── RETOUR (version batch) ─────────────────────────────────────────────────
        self.iteration += 1

        IndexRecommendation = df_Candidate.iloc[best_candidate_iloc_batch].index.to_list()

        print("WiGS_SAC_Selector IndexRecommendation")

        return {"IndexRecommendation": IndexRecommendation, "w_x": w_x}
