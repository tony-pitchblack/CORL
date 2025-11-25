#!/usr/bin/env python3

import os
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import minari
from minari.utils import get_normalized_score
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import mlflow
import mlflow.data
from dotenv import load_dotenv
import gymnasium as gym
import pandas as pd

# DATASET_ID = "D4RL/door/human-v2"
DATASET_ID = "minigrid/BabyAI-GoToObjS4/optimal-fullobs-v0"
# DATASET_ID = "minigrid/BabyAI-GoToObj/optimal-fullobs-v0"

TensorBatch = List[torch.Tensor]


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, state_mean: np.ndarray, state_std: np.ndarray):
        super().__init__(env)
        self._state_mean = state_mean
        self._state_std = state_std

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self._state_mean) / self._state_std


class ScaledRewardWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return self._scale * reward


@dataclass
class TrainConfig:
    # Experiment
    dataset_id: str = DATASET_ID
    download: bool = True
    device: str = "cuda:1"
    seed: int = 0
    num_train_steps: int = int(1e6)
    batch_size: int = 256
    buffer_size: int = 2_000_000
    frac: float = 1.0
    max_traj_len: int = 1000
    discount: float = 0.99
    normalize: bool = True
    # Evaluation
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    # MLflow logging
    experiment_name: str = f"BC-{DATASET_ID}"
    run_name: str = "BC"


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: np.ndarray = 0.0,
    state_std: np.ndarray = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    env = NormalizedObservationWrapper(env, state_mean, state_std)
    if reward_scale != 1.0:
        env = ScaledRewardWrapper(env, reward_scale)
    return env


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories: List[List[int]] = []
    returns: List[float] = []
    cur_ids: List[int] = []
    cur_return = 0.0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * float(reward)
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0.0
            reward_scale = 1.0
    sort_ord = np.argsort(np.array(returns))[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]
    order: List[int] = []
    for i in top_trajs:
        order += ids_by_trajectories[int(i)]
    order_arr = np.asarray(order, dtype=np.int64)
    dataset["observations"] = dataset["observations"][order_arr]
    dataset["actions"] = dataset["actions"][order_arr]
    dataset["next_observations"] = dataset["next_observations"][order_arr]
    dataset["rewards"] = dataset["rewards"][order_arr]
    dataset["terminals"] = dataset["terminals"][order_arr]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size = n_transitions
        self._pointer = n_transitions

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.reset(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def make_minari_evaluator(
    env: gym.Env, n_episodes: int, seed: int, device: str
):
    @torch.no_grad()
    def _eval_actor(actor: nn.Module) -> np.ndarray:
        env.reset(seed=seed)
        actor.eval()
        episode_rewards = []
        for _ in range(n_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = actor.act(state, device)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            episode_rewards.append(episode_reward)
        actor.train()
        return np.asarray(episode_rewards)

    return _eval_actor


def setup_mlflow():
    """Setup MLflow tracking with configuration from .env file."""
    load_dotenv()
    mlflow_host = os.getenv("MLFLOW_HOST", "localhost")
    mlflow_port = os.getenv("MLFLOW_PORT", "5001")
    tracking_uri = f"http://{mlflow_host}:{mlflow_port}"
    mlflow.set_tracking_uri(tracking_uri)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cuda") -> np.ndarray:
        state_t = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state_t).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cuda",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount
        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict


def minari_dataset_to_d4rl(dataset) -> Dict[str, np.ndarray]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    next_observations: List[np.ndarray] = []
    terminals: List[np.ndarray] = []
    for episode in dataset.iterate_episodes():
        obs = episode.observations
        act = episode.actions
        rew = episode.rewards
        term = episode.terminations
        trunc = episode.truncations
        done = np.logical_or(term, trunc).astype(np.float32)
        observations.append(obs[:-1])
        next_observations.append(obs[1:])
        actions.append(act)
        rewards.append(rew.astype(np.float32))
        terminals.append(done.astype(np.float32))
    observations_arr = np.concatenate(observations, axis=0)
    actions_arr = np.concatenate(actions, axis=0)
    rewards_arr = np.concatenate(rewards, axis=0)
    next_observations_arr = np.concatenate(next_observations, axis=0)
    terminals_arr = np.concatenate(terminals, axis=0)
    return {
        "observations": observations_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "next_observations": next_observations_arr,
        "terminals": terminals_arr,
    }


@pyrallis.wrap()
def train(config: TrainConfig):
    # Setup MLflow
    setup_mlflow()
    mlflow.set_experiment(config.experiment_name)
    algo_name = config.run_name
    dataset_name = config.dataset_id.replace("/", "-")
    random_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    run_name = f"{algo_name}-{dataset_name}-{random_suffix}"
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        mlflow.log_params({
            "dataset_id": config.dataset_id,
            "seed": config.seed,
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "frac": config.frac,
            "max_traj_len": config.max_traj_len,
            "discount": config.discount,
            "normalize": config.normalize,
            "eval_freq": config.eval_freq,
            "n_episodes": config.n_episodes,
        })

        dataset = minari.load_dataset(config.dataset_id, download=config.download)
        env = dataset.recover_environment()
        env_name = getattr(getattr(env, "spec", None), "id", None) or config.dataset_id
        env_df = pd.DataFrame(columns=[env_name])
        env_dataset = mlflow.data.from_pandas(
            env_df,
            source=env_name,
            name=env_name,
        )
        mlflow.log_input(env_dataset, context="environment")
        set_seed(config.seed, env)
        state_dim = int(np.prod(dataset.observation_space.shape))
        action_dim = int(np.prod(dataset.action_space.shape))
        d4rl_dataset = minari_dataset_to_d4rl(dataset)
        keep_best_trajectories(
            d4rl_dataset,
            frac=config.frac,
            discount=config.discount,
            max_episode_steps=config.max_traj_len,
        )
        if config.normalize:
            state_mean, state_std = compute_mean_std(
                d4rl_dataset["observations"], eps=1e-3
            )
            d4rl_dataset["observations"] = normalize_states(
                d4rl_dataset["observations"], state_mean, state_std
            )
            d4rl_dataset["next_observations"] = normalize_states(
                d4rl_dataset["next_observations"], state_mean, state_std
            )
            env = wrap_env(env, state_mean, state_std)
        else:
            state_mean = np.zeros(state_dim)
            state_std = np.ones(state_dim)
        eval_actor = make_minari_evaluator(
            env=env,
            n_episodes=config.n_episodes,
            seed=config.seed,
            device=config.device,
        )
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=config.buffer_size,
            device=config.device,
        )
        replay_buffer.load_d4rl_dataset(d4rl_dataset)
        max_action = float(dataset.action_space.high[0])
        actor = Actor(state_dim, action_dim, max_action).to(config.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        trainer = BC(
            max_action=max_action,
            actor=actor,
            actor_optimizer=actor_optimizer,
            discount=config.discount,
            device=config.device,
        )
        for step in trange(config.num_train_steps, ncols=80):
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            # Log training metrics
            for key, value in log_dict.items():
                mlflow.log_metric(key, value, step=step)
            
            # Evaluate episode
            if (step + 1) % config.eval_freq == 0:
                eval_scores = eval_actor(actor)
                eval_score = eval_scores.mean()
                mlflow.log_metric("eval_score", eval_score, step=step)
                mlflow.log_metric("eval_score_std", eval_scores.std(), step=step)
                normalized_eval_score = get_normalized_score(
                    dataset, eval_scores
                ) * 100.0
                mlflow.log_metric(
                    "normalized_score", normalized_eval_score.mean(), step=step
                )


if __name__ == "__main__":
    train()


