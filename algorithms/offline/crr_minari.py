#!/usr/bin/env python3

import os
import random
import string
from copy import deepcopy
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

DATASET_ID = "D4RL/door/human-v2"

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
    hidden_dim: int = 256
    critic_lr: float = 3e-4
    tau: float = 5e-3
    crr_beta: float = 1.0
    crr_weight_type: str = "binary"  # "binary" or "exp"
    crr_max_weight: float = 20.0
    # Evaluation
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    # MLflow logging
    experiment_name: str = "CORL-Minari"
    run_name: str = "CRR"


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


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class CRR:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 5e-3,
        beta: float = 1.0,
        weight_type: str = "binary",
        max_weight: float = 20.0,
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_optimizer = critic_1_optimizer
        self.target_critic_1 = deepcopy(critic_1)
        self.critic_2 = critic_2
        self.critic_2_optimizer = critic_2_optimizer
        self.target_critic_2 = deepcopy(critic_2)
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.weight_type = weight_type
        self.max_weight = max_weight

    def _critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self.actor(next_states)
            q_next_1 = self.target_critic_1(next_states, next_actions)
            q_next_2 = self.target_critic_2(next_states, next_actions)
            q_next = torch.min(q_next_1, q_next_2)
            target_q = rewards + self.discount * (1.0 - dones) * q_next
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        return loss

    def _actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            pi_actions = self.actor(states)
            q_pi_1 = self.critic_1(states, pi_actions)
            q_pi_2 = self.critic_2(states, pi_actions)
            v = torch.min(q_pi_1, q_pi_2)
            q_b_1 = self.critic_1(states, actions)
            q_b_2 = self.critic_2(states, actions)
            q_b = torch.min(q_b_1, q_b_2)
            adv = q_b - v
            if self.weight_type == "binary":
                weights = (adv > 0.0).float()
            else:
                weights = torch.exp(adv / self.beta)
                weights = torch.clamp(weights, max=self.max_weight)
        pi_pred = self.actor(states)
        mse = F.mse_loss(pi_pred, actions, reduction="none").mean(dim=-1, keepdim=True)
        loss = (weights * mse).mean()
        return loss, weights.mean().item()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        actor_loss, weight_mean = self._actor_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "weight_mean": weight_mean,
        }


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
    setup_mlflow()
    mlflow.set_experiment(config.experiment_name)
    algo_name = config.run_name
    dataset_name = config.dataset_id.replace("/", "-")
    random_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    run_name = f"{algo_name}-{dataset_name}-{random_suffix}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dataset_id": config.dataset_id,
            "seed": config.seed,
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "buffer_size": config.buffer_size,
            "frac": config.frac,
            "max_traj_len": config.max_traj_len,
            "discount": config.discount,
            "normalize": config.normalize,
            "hidden_dim": config.hidden_dim,
            "critic_lr": config.critic_lr,
            "tau": config.tau,
            "crr_beta": config.crr_beta,
            "crr_weight_type": config.crr_weight_type,
            "crr_max_weight": config.crr_max_weight,
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

        critic_1 = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
        critic_2 = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.critic_lr)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.critic_lr)

        trainer = CRR(
            actor=actor,
            actor_optimizer=actor_optimizer,
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            discount=config.discount,
            tau=config.tau,
            beta=config.crr_beta,
            weight_type=config.crr_weight_type,
            max_weight=config.crr_max_weight,
        )

        for step in trange(config.num_train_steps, ncols=80):
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            for key, value in log_dict.items():
                mlflow.log_metric(key, value, step=step)

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


