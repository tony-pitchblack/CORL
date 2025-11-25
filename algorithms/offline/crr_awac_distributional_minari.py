#!/usr/bin/env python3

import os
import random
import string
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import minari
from minari.utils import get_normalized_score
import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import pyrallis
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

DATASET_ID = "D4RL/door/human-v2"

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    dataset_id: str = DATASET_ID
    download: bool = True
    seed: int = 42
    deterministic_torch: bool = False
    device: str = "cuda:2"
    num_train_steps: int = 1_000_000
    batch_size: int = 256
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    crr_beta: float = 1.0
    crr_weight_type: str = "binary"  # "binary" or "exp"
    crr_max_weight: float = 20.0
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    test_seed: int = 69
    experiment_name: str = "CORL-Minari"
    run_name: str = "CRR-AWAC-Distributional"


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
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

    def load(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ):
        if self._size != 0:
            raise ValueError("Replay buffer is not empty")
        n_transitions = observations.shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load"
            )
        self._states[:n_transitions] = self._to_tensor(observations)
        self._actions[:n_transitions] = self._to_tensor(actions)
        self._rewards[:n_transitions] = self._to_tensor(rewards)
        self._next_states[:n_transitions] = self._to_tensor(next_observations)
        self._dones[:n_transitions] = self._to_tensor(dones)
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


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        return policy.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        return action_t[0].cpu().numpy()


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_atoms: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_atoms),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class CRRDistributional:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        beta: float,
        weight_type: str,
        max_weight: float,
        n_atoms: int,
        v_min: float,
        v_max: float,
        device: str,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer
        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)
        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)
        self._gamma = gamma
        self._tau = tau
        self._beta = beta
        self._weight_type = weight_type
        self._max_weight = max_weight
        self._n_atoms = n_atoms
        self._v_min = v_min
        self._v_max = v_max
        support = torch.linspace(v_min, v_max, n_atoms, device=device)
        self._support = support
        self._delta_z = (v_max - v_min) / float(n_atoms - 1)

    def _logits_to_q(self, logits: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=-1)
        return (prob * self._support.view(1, -1)).sum(dim=-1, keepdim=True)

    def _project_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        prob_next = F.softmax(next_logits, dim=-1)
        rewards = rewards.expand(-1, self._n_atoms)
        dones = dones.expand(-1, self._n_atoms)
        support = self._support.view(1, -1)
        tz = rewards + (1.0 - dones) * self._gamma * support
        tz = tz.clamp(self._v_min, self._v_max)
        b = (tz - self._v_min) / self._delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l = l.clamp(0, self._n_atoms - 1)
        u = u.clamp(0, self._n_atoms - 1)
        offset = (
            torch.arange(batch_size, device=rewards.device)
            .unsqueeze(1)
            .expand(batch_size, self._n_atoms)
            * self._n_atoms
        )
        proj_dist = torch.zeros_like(prob_next)
        proj_dist.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (prob_next * (u.float() - b)).view(-1),
        )
        proj_dist.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (prob_next * (b - l.float())).view(-1),
        )
        return proj_dist

    def _dist_loss(
        self,
        logits: torch.Tensor,
        target_prob: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        return -(target_prob * log_prob).sum(dim=-1).mean()

    def _critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)
            logits1_next = self._target_critic_1(next_states, next_actions)
            logits2_next = self._target_critic_2(next_states, next_actions)
            q1_next = self._logits_to_q(logits1_next)
            q2_next = self._logits_to_q(logits2_next)
            use_1 = (q1_next <= q2_next).float()
            logits_next = logits1_next * use_1 + logits2_next * (1.0 - use_1)
            target_prob = self._project_distribution(rewards, dones, logits_next)
        logits1 = self._critic_1(states, actions)
        logits2 = self._critic_2(states, actions)
        loss1 = self._dist_loss(logits1, target_prob)
        loss2 = self._dist_loss(logits2, target_prob)
        return loss1 + loss2

    def _actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            pi_actions, _ = self._actor(states)
            logits_pi_1 = self._critic_1(states, pi_actions)
            logits_pi_2 = self._critic_2(states, pi_actions)
            v1 = self._logits_to_q(logits_pi_1)
            v2 = self._logits_to_q(logits_pi_2)
            v = torch.min(v1, v2)
            logits_b_1 = self._critic_1(states, actions)
            logits_b_2 = self._critic_2(states, actions)
            q_b1 = self._logits_to_q(logits_b_1)
            q_b2 = self._logits_to_q(logits_b_2)
            q_b = torch.min(q_b1, q_b2)
            adv = q_b - v
            if self._weight_type == "binary":
                weights = (adv > 0.0).float()
            else:
                weights = torch.exp(adv / self._beta)
                weights = torch.clamp(weights, max=self._max_weight)
        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        actor_loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, state_mean: np.ndarray, state_std: np.ndarray):
        super().__init__(env)
        self._state_mean = state_mean
        self._state_std = state_std

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self._state_mean) / self._state_std


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(env: gym.Env, state_mean: np.ndarray, state_std: np.ndarray) -> gym.Env:
    return NormalizedObservationWrapper(env, state_mean, state_std)


def make_minari_evaluator(env: gym.Env, n_episodes: int, seed: int, device: str):
    @torch.no_grad()
    def _eval_actor(actor: Actor) -> np.ndarray:
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


def minari_dataset_to_transitions(dataset) -> Dict[str, np.ndarray]:
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    for episode in dataset.iterate_episodes():
        obs = episode.observations
        act = episode.actions
        rew = episode.rewards
        term = episode.terminations
        trunc = episode.truncations
        done = np.logical_or(term, trunc)
        observations.append(obs[:-1])
        next_observations.append(obs[1:])
        actions.append(act)
        rewards.append(rew.reshape(-1, 1) if rew.ndim == 1 else rew)
        dones.append(done.reshape(-1, 1) if done.ndim == 1 else done)
    observations_arr = np.concatenate(observations, axis=0)
    actions_arr = np.concatenate(actions, axis=0)
    rewards_arr = np.concatenate(rewards, axis=0)
    next_observations_arr = np.concatenate(next_observations, axis=0)
    dones_arr = np.concatenate(dones, axis=0)
    return {
        "observations": observations_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "next_observations": next_observations_arr,
        "dones": dones_arr,
    }


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
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
            "deterministic_torch": config.deterministic_torch,
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "tau": config.tau,
            "crr_beta": config.crr_beta,
            "crr_weight_type": config.crr_weight_type,
            "crr_max_weight": config.crr_max_weight,
            "n_atoms": config.n_atoms,
            "v_min": config.v_min,
            "v_max": config.v_max,
            "eval_frequency": config.eval_frequency,
            "n_test_episodes": config.n_test_episodes,
            "test_seed": config.test_seed,
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

        state_dim = int(np.prod(dataset.observation_space.shape))
        action_dim = int(np.prod(dataset.action_space.shape))
        transitions = minari_dataset_to_transitions(dataset)
        state_mean, state_std = compute_mean_std(
            transitions["observations"], eps=1e-3
        )
        transitions["observations"] = normalize_states(
            transitions["observations"], state_mean, state_std
        )
        transitions["next_observations"] = normalize_states(
            transitions["next_observations"], state_mean, state_std
        )
        env = wrap_env(env, state_mean=state_mean, state_std=state_std)
        n_transitions = transitions["observations"].shape[0]
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=n_transitions,
            device=config.device,
        )
        replay_buffer.load(
            transitions["observations"],
            transitions["actions"],
            transitions["rewards"],
            transitions["next_observations"],
            transitions["dones"],
        )

        min_action = float(dataset.action_space.low.min())
        max_action = float(dataset.action_space.high.max())
        actor_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": config.hidden_dim,
            "min_action": min_action,
            "max_action": max_action,
        }
        actor = Actor(**actor_kwargs).to(config.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)

        critic_1 = DistributionalCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            n_atoms=config.n_atoms,
        ).to(config.device)
        critic_2 = DistributionalCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            n_atoms=config.n_atoms,
        ).to(config.device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(), lr=config.learning_rate
        )
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(), lr=config.learning_rate
        )

        algo = CRRDistributional(
            actor=actor,
            actor_optimizer=actor_optimizer,
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            gamma=config.gamma,
            tau=config.tau,
            beta=config.crr_beta,
            weight_type=config.crr_weight_type,
            max_weight=config.crr_max_weight,
            n_atoms=config.n_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            device=config.device,
        )

        eval_actor = make_minari_evaluator(
            env=env,
            n_episodes=config.n_test_episodes,
            seed=config.test_seed,
            device=config.device,
        )

        for step in trange(config.num_train_steps, ncols=80):
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            update_result = algo.update(batch)
            for key, value in update_result.items():
                mlflow.log_metric(key, value, step=step)
            if (step + 1) % config.eval_frequency == 0:
                eval_scores = eval_actor(actor)
                eval_score = eval_scores.mean()
                mlflow.log_metric("eval_score", eval_score, step=step)
                mlflow.log_metric("eval_score_std", eval_scores.std(), step=step)
                normalized_eval_scores = get_normalized_score(
                    dataset, eval_scores
                ) * 100.0
                mlflow.log_metric(
                    "normalized_score",
                    normalized_eval_scores.mean(),
                    step=step,
                )


if __name__ == "__main__":
    train()


