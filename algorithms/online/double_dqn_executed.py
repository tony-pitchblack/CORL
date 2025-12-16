# %%
ALGO_NAME = "Double_DQN"

# %%
# CONFIG: minigrid/BabyAI-GoToObj/optimal-fullobs-v0 | default

ENVIRONMENT = "minigrid/BabyAI-GoToObj/optimal-fullobs-v0"

MINIBATCH_SIZE = 64
MAX_STEPS = 200_000
EVAL_EVERY_N_STEPS = 10_000
N_EVAL_EPISODES = 100
COPY_WEIGHTS_EVERY_N_STEPS = 5_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 20_000
GAMMA = 0.99
LR = 1e-3
WINDOW_SIZE_REWARD = 100
WINDOW_SIZE_LOSS = 100
MIN_BUFFER_SIZE = 5_000
NET_ARCH = "cnn-minimal"
VIDEO_MACRO_BLOCK_SIZE=16

# %%
# # CONFIG: MiniGrid-Empty-5x5-v0 | default

# ENVIRONMENT = "MiniGrid-Empty-5x5-v0"

# MINIBATCH_SIZE = 64
# MAX_STEPS = 64_000
# EVAL_EVERY_N_STEPS = 2000
# N_EVAL_EPISODES = 100
# COPY_WEIGHTS_EVERY_N_STEPS = 1000
# EPS_START = 1.0
# EPS_END = 0.01
# EPS_DECAY_STEPS = 40_000
# GAMMA = 0.9
# LR = 1e-3
# WINDOW_SIZE_REWARD = 100
# WINDOW_SIZE_LOSS = 100
# MIN_BUFFER_SIZE = 64
# NET_ARCH = "cnn-minimal"
# VIDEO_MACRO_BLOCK_SIZE = 10

# %%
# # CONFIG: MiniGrid-Empty-5x5-v0 | lower eval/copy/eps intervals

# ENVIRONMENT = "MiniGrid-Empty-5x5-v0"

# MINIBATCH_SIZE = 64
# MAX_STEPS = 64_000
# EVAL_EVERY_N_STEPS = 2000
# N_EVAL_EPISODES = 100
# COPY_WEIGHTS_EVERY_N_STEPS = 1_000
# EPS_START = 1.0
# EPS_END = 0.05
# EPS_DECAY_STEPS = 40_000
# GAMMA = 0.9
# LR = 1e-3
# WINDOW_SIZE_REWARD = 100
# WINDOW_SIZE_LOSS = 100
# MIN_BUFFER_SIZE = 1_000
# NET_ARCH = "cnn-minimal"
# VIDEO_MACRO_BLOCK_SIZE = 10

# %%
# # CONFIG: MiniGrid-Empty-5x5-v0 | use mlp net arch

# ENVIRONMENT = "MiniGrid-Empty-5x5-v0"

# MINIBATCH_SIZE = 64
# MAX_STEPS = 64_000
# EVAL_EVERY_N_STEPS = 2000
# N_EVAL_EPISODES = 100
# COPY_WEIGHTS_EVERY_N_STEPS = 1_000
# EPS_START = 1.0
# EPS_END = 0.05
# EPS_DECAY_STEPS = 40_000
# GAMMA = 0.9
# LR = 1e-3
# WINDOW_SIZE_REWARD = 100
# WINDOW_SIZE_LOSS = 100
# MIN_BUFFER_SIZE = 1_000
# NET_ARCH = "mlp-minimal"
# # NET_ARCH = "cnn-minimal"
# VIDEO_MACRO_BLOCK_SIZE = 10

# %%
# # CONFIG: CartPole-v1 | default

# ENVIRONMENT = "CartPole-v1"

# MINIBATCH_SIZE = 64
# MAX_STEPS = 50_000
# EVAL_EVERY_N_STEPS = 1_000
# N_EVAL_EPISODES = 10
# COPY_WEIGHTS_EVERY_N_STEPS = 500
# EPS_START = 1.0
# EPS_END = 0.05
# EPS_DECAY_STEPS = 30_000
# GAMMA = 0.9
# LR = 1e-3
# WINDOW_SIZE_REWARD = 100
# WINDOW_SIZE_LOSS = 100
# MIN_BUFFER_SIZE = 2_000
# VIDEO_MACRO_BLOCK_SIZE = 10
# NET_ARCH = "mlp-minimal"

# %%
import minari

def create_env(*args, **kwargs):
    env_id = args[0] if args else ENVIRONMENT
    try:
        dataset = minari.load_dataset(env_id, download=True)
        return dataset.recover_environment(**kwargs)
    except ValueError:
        import gymnasium as gym
        import minigrid
        return gym.make(*args, **kwargs)


# %%
from torch import nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_obs(obs):
    x = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if x.ndim >= 3:
        if x.ndim == 3:
            if x.shape[-1] in (1, 3, 4):
                x = x.permute(2, 0, 1)
        elif x.ndim == 4:
            if x.shape[-1] in (1, 3, 4):
                x = x.permute(0, 3, 1, 2)
        x = x / 255.0
    return x

class QNet(nn.Module):
    def __init__(self, num_actions, input_shape, arch="cnn-minimal"):
        super().__init__()
        self.arch = arch
        if arch == "cnn-minimal":
            if len(input_shape) != 3:
                raise ValueError(f"cnn-minimal expects 3D input, got shape {input_shape}")
            c, h, w = input_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.flatten = nn.Flatten()
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                conv_out = self.flatten(self.conv(dummy))
                linear_input_size = conv_out.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(linear_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        elif arch == "mlp-minimal":
            input_dim = int(np.prod(input_shape))
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        else:
            raise ValueError(f"Unknown NET_ARCH: {arch}")

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.arch == "cnn-minimal":
            x = self.conv(x)
            x = self.flatten(x)
            x = self.fc(x)
        else:
            x = self.mlp(x)
        return x

# %%
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, device=device):
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __len__(self):
        return len(self.states)

    def add_transition(self, s, a, r, s_next, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_next)
        self.dones.append(done)

    def sample_minibatch(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size, replace=False)

        states = torch.stack([self.states[i] for i in indices]).to(self.device)
        next_states = torch.stack([self.next_states[i] for i in indices]).to(self.device)
        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor([self.dones[i] for i in indices], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

# %%
import os
from pathlib import Path
import torch
import imageio.v2 as imageio
from tqdm import trange
import numpy as np

def _find_project_root(start_path: Path | None = None) -> Path:
    base = start_path or Path(os.getcwd()).resolve()
    for path in [base, *base.parents]:
        if (path / ".git").exists() or (path / "README.md").exists():
            return path
    return base


def evaluate(policy, n_envs=100, log_date=None, log_time=None, log_step=0, q_lower=0.1, q_upper=0.9):
    """
    policy: Function mapping obs -> action, or a PyTorch model with .eval()
    n_envs: Number of evaluation episodes
    log_date, log_time, log_step: Logging identifiers
    q_lower, q_upper: Quantiles (floats)
    """
    import datetime
    if log_date is None or log_time is None:
        now = datetime.datetime.now()
        if log_date is None:
            log_date = now.strftime("%Y-%m-%d")
        if log_time is None:
            log_time = now.strftime("%H-%M-%S")
    rewards = []
    episodes = []
    action_sequences = []

    for i in trange(n_envs, desc="Evaluating episodes"):
        env = create_env(ENVIRONMENT, render_mode="rgb_array")
        obs, info = env.reset()
        episode_reward = 0
        frames = []
        actions = []
        terminated = False
        truncated = False

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not (terminated or truncated):
            if callable(getattr(policy, "eval", None)):
                policy.eval()
            with torch.no_grad():
                if isinstance(obs, dict) and "image" in obs:
                    obs_tensor = preprocess_obs(obs["image"]).unsqueeze(0)
                else:
                    obs_tensor = preprocess_obs(obs).unsqueeze(0)
                qvals = policy(obs_tensor)
                action = qvals.argmax().item()
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        rewards.append(episode_reward)
        episodes.append(frames)
        action_sequences.append(actions)
        env.close()

    rewards = np.array(rewards)
    sort_idx = np.argsort(rewards)
    idx_lower = sort_idx[int(q_lower * n_envs)]
    idx_upper = sort_idx[int(q_upper * n_envs)]
    idx_mean = sort_idx[int(0.5 * n_envs)]
    quantiles = [
        (q_lower, idx_lower),
        (0.5, idx_mean),
        (q_upper, idx_upper)
    ]

    mean_reward = float(np.mean(rewards))

    project_root = _find_project_root()
    output_dir_path = project_root / "videos" / "online" / ALGO_NAME / ENVIRONMENT / f"{log_date}_{log_time}" / f"steps={log_step}_reward@{n_envs}={mean_reward:.4f}"
    output_dir = str(output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    for quantile, idx in quantiles:
        frames = episodes[idx]
        quantile_reward = rewards[idx]
        filename = os.path.join(
            output_dir,
            f"q={quantile:.2f}_reward={quantile_reward:.4f}.mp4"
        )
        if os.path.exists(filename):
            os.remove(filename)
        try:
            imageio.mimsave(filename, frames, fps=30, macro_block_size=VIDEO_MACRO_BLOCK_SIZE)
        except Exception:
            pass

    return {
        "mean_reward": mean_reward,
        "rewards": rewards,
        "quantile_indices": {str(q): int(i) for q, i in quantiles},
        "output_dir": output_dir,
    }

# %%
# qnet = QNet(
#     num_actions=env.action_space.n
# )

# evaluate(qnet)

# %%
import os
import mlflow
from pathlib import Path
import pandas as pd


def load_env_vars(env_path):
    env = {}
    path = Path(env_path)
    resolved_path = path.resolve()
    print(f"Looking for .env in {resolved_path}...")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


papermill_input = os.environ.get("PAPERMILL_INPUT_PATH")
if papermill_input:
    notebook_dir = Path(papermill_input).resolve().parent
    env_path = notebook_dir.parent / ".env"
else:
    env_path = Path.cwd() / ".env"

_env_vars = load_env_vars(env_path)
_mlflow_server = _env_vars["MLFLOW_HOST"]
_mlflow_port = _env_vars["MLFLOW_PORT"]

mlflow_logging_uri = f"http://{_mlflow_server}:{_mlflow_port}"
mlflow.set_tracking_uri(mlflow_logging_uri)


# %%
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque
import datetime
import string
import random

# Log hparams as config to mlflow
config = {
    "minibatch_size": MINIBATCH_SIZE,
    "max_steps": MAX_STEPS,
    "gamma": GAMMA,
    "lr": LR,
    "eps_start": EPS_START,
    "eps_end": EPS_END,
    "eps_decay_steps": EPS_DECAY_STEPS,
    "copy_target_every": COPY_WEIGHTS_EVERY_N_STEPS,
    "eval_every": EVAL_EVERY_N_STEPS,
    "reward_window": WINDOW_SIZE_REWARD,
    "loss_window": WINDOW_SIZE_LOSS,
    "min_buffer_size": MIN_BUFFER_SIZE,
    "n_eval_episodes": N_EVAL_EPISODES,
    "net_arch": NET_ARCH,
}

env = create_env(ENVIRONMENT, render_mode="rgb_array")
sample_obs, _ = env.reset()
if isinstance(sample_obs, dict) and "image" in sample_obs:
    sample_tensor = preprocess_obs(sample_obs["image"])
else:
    sample_tensor = preprocess_obs(sample_obs)
input_shape = sample_tensor.shape
replay_buffer = ReplayBuffer(device=device)
qnet_action = QNet(num_actions=env.action_space.n, input_shape=input_shape, arch=NET_ARCH).to(device)
qnet_target = QNet(num_actions=env.action_space.n, input_shape=input_shape, arch=NET_ARCH).to(device)
qnet_target.load_state_dict(qnet_action.state_dict())
optimizer = torch.optim.Adam(qnet_action.parameters(), lr=LR)

sliding_rewards = deque(maxlen=WINDOW_SIZE_REWARD)
sliding_losses = deque(maxlen=WINDOW_SIZE_LOSS)
n_steps = 0
now = datetime.datetime.now()
log_time = now.strftime("%H-%M-%S")
log_date = now.strftime("%Y-%m-%d")

mlflow.set_experiment(f"{ALGO_NAME}_{ENVIRONMENT}")
if mlflow.active_run() is not None:
    mlflow.end_run()

run_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
run_name = f"{ALGO_NAME}-{run_suffix}"
mlflow.start_run(run_name=run_name)

env_df = pd.DataFrame(columns=["environment"])
env_dataset = mlflow.data.from_pandas(
    env_df,
    source=ENVIRONMENT,
    name=f"env-{ENVIRONMENT}",
)
mlflow.log_input(env_dataset)

mlflow.log_dict(config, "config.yaml")

pbar = tqdm(total=MAX_STEPS, desc="Steps", leave=True)
while n_steps < MAX_STEPS:
    done = False
    episode_reward = 0
    state, info = env.reset()
    if isinstance(state, dict) and "image" in state:
        state_tensor = preprocess_obs(state["image"])
    else:
        state_tensor = preprocess_obs(state)

    while not done and n_steps < MAX_STEPS:
        with torch.no_grad():
            EPS = EPS_END + (EPS_START - EPS_END) * np.exp(-n_steps / EPS_DECAY_STEPS)
            if np.random.random() < EPS:
                action = np.random.choice(env.action_space.n)
            else:
                qvalues = qnet_action(state_tensor.unsqueeze(0))
                action = qvalues.argmax().item()

        # Log epsilon for this step
        mlflow.log_metric("epsilon", EPS, step=n_steps)

        state_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        if isinstance(state_next, dict) and "image" in state_next:
            state_next_tensor = preprocess_obs(state_next["image"])
        else:
            state_next_tensor = preprocess_obs(state_next)
        replay_buffer.add_transition(state_tensor, action, reward, state_next_tensor, done)

        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample_minibatch(MINIBATCH_SIZE)
            with torch.no_grad():
                qnet_action_next = qnet_action(next_states_b).argmax(dim=-1)
                double_targets = torch.take_along_dim(
                    qnet_target(next_states_b),
                    qnet_action_next.unsqueeze(1),
                    dim=1
                ).squeeze(1)
                targets_b = rewards_b + (1 - dones_b) * GAMMA * double_targets

            qvalues_all = qnet_action(states_b)
            qvalues_b = torch.take_along_dim(qvalues_all, actions_b.unsqueeze(1), dim=1).squeeze(1)

            loss = F.mse_loss(qvalues_b, targets_b)
            sliding_losses.append(loss.item())

            # Log smoothed loss if we have at least 1 value in the window
            if len(sliding_losses) > 0:
                avg_loss = sum(sliding_losses) / len(sliding_losses)
                mlflow.log_metric(f"avg_loss:window_{WINDOW_SIZE_LOSS}", avg_loss, step=n_steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = state_next
        state_tensor = state_next_tensor
        n_steps += 1
        pbar.update(1)

        if n_steps % COPY_WEIGHTS_EVERY_N_STEPS == 0:
            qnet_target.load_state_dict(qnet_action.state_dict())

        if n_steps % EVAL_EVERY_N_STEPS == 0:
            eval_result = evaluate(qnet_action, log_time=log_time, log_date=log_date, log_step=n_steps, n_envs=N_EVAL_EPISODES)
            print(f"\nEvaluation at step {n_steps}: mean_reward={eval_result['mean_reward']:.3f}")
            mlflow.log_metric(f"eval_reward:window_{N_EVAL_EPISODES}", eval_result["mean_reward"], step=n_steps)

    sliding_rewards.append(episode_reward)
    avg_reward = sum(sliding_rewards) / len(sliding_rewards) if sliding_rewards else 0.0
    pbar.set_postfix({f"train_reward:window_{WINDOW_SIZE_REWARD}": avg_reward})
    mlflow.log_metric(f"train_reward:window_{WINDOW_SIZE_REWARD}", avg_reward, step=n_steps)

pbar.close()
if mlflow.active_run() is not None:
    mlflow.end_run()

# %%



