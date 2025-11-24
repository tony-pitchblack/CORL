#!/usr/bin/env python3

import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import minari
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tqdm import trange
import mlflow
from dotenv import load_dotenv
import gymnasium as gym


TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    dataset_id: str = "D4RL/door/human-v2"
    download: bool = True
    seed: int = 42
    deterministic_torch: bool = False
    device: str = "gpu"
    num_train_steps: int = 1_000_000
    batch_size: int = 256
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    awac_lambda: float = 1.0
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    test_seed: int = 69
    # MLflow logging
    experiment_name: str = "CORL-Minari"
    run_name: str = "AWAC"


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
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

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
        action = action_t[0].cpu().numpy()
        return action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
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
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
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
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )
            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )
        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)
            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next
        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)
        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)
        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(env: gym.Env, state_mean: np.ndarray, state_std: np.ndarray) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


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
    """Setup MLflow tracking with configuration from .env file."""
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

    # Setup MLflow
    setup_mlflow()
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name):
        # Log configuration
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
            "awac_lambda": config.awac_lambda,
                "eval_frequency": config.eval_frequency,
                "n_test_episodes": config.n_test_episodes,
                "test_seed": config.test_seed,
        })

        dataset = minari.load_dataset(config.dataset_id, download=config.download)
        env = dataset.recover_environment()
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
        actor_critic_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": config.hidden_dim,
            "min_action": min_action,
            "max_action": max_action,
        }
        actor = Actor(**actor_critic_kwargs)
        actor.to(config.device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
        critic_1 = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim)
        critic_2 = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim=config.hidden_dim)
        critic_1.to(config.device)
        critic_2.to(config.device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)
        awac = AdvantageWeightedActorCritic(
            actor=actor,
            actor_optimizer=actor_optimizer,
            critic_1=critic_1,
            critic_1_optimizer=critic_1_optimizer,
            critic_2=critic_2,
            critic_2_optimizer=critic_2_optimizer,
            gamma=config.gamma,
            tau=config.tau,
            awac_lambda=config.awac_lambda,
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
            update_result = awac.update(batch)
            # Log training metrics
            for key, value in update_result.items():
                mlflow.log_metric(key, value, step=step)
            if (step + 1) % config.eval_frequency == 0:
                eval_scores = eval_actor(actor)
                eval_score = eval_scores.mean()
                mlflow.log_metric("eval_score", eval_score, step=step)
                mlflow.log_metric("eval_score_std", eval_scores.std(), step=step)
                if hasattr(env, "get_normalized_score"):
                    normalized_eval_scores = env.get_normalized_score(eval_scores) * 100.0
                    mlflow.log_metric(
                        "normalized_score",
                        normalized_eval_scores.mean(),
                        step=step,
                    )


if __name__ == "__main__":
    train()


