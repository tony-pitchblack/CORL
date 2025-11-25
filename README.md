# CORL (Clean Offline Reinforcement Learning)

[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/vladkurenkov/status/1669361090550177793)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07105-b31b1b.svg)](https://arxiv.org/abs/2210.07105)
[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


🧵 CORL is an Offline Reinforcement Learning library that provides high-quality and easy-to-follow single-file implementations of SOTA ORL algorithms. Each implementation is backed by a research-friendly codebase, allowing you to run or tune thousands of experiments. Heavily inspired by [cleanrl](https://github.com/vwxyzjn/cleanrl) for online RL, check them out too!

## MLflow Setup (for Minari-based algorithms)

Minari-based algorithms (`algorithms/offline/*_minari.py`) use MLflow for experiment tracking. To use them:

1. Start MLflow server (using values from `.env` file):
```bash
source .env && mlflow server --host $MLFLOW_HOST --port $MLFLOW_PORT
```

   Or start in tmux for persistent background running:
```bash
tmux kill-session -t mlflow 2>/dev/null || true; tmux new -s mlflow zsh -c 'source .env && mlflow server --host $MLFLOW_HOST --port $MLFLOW_PORT; exec zsh'
```

```zsh
tmux kill-session -t mlflow 2>/dev/null || true; tmux new -s mlflow zsh -c 'source .env && mlflow server --host $MLFLOW_HOST --port $MLFLOW_PORT; exec zsh'
```

## Training Examples

Launch training scripts with tmux for persistent background execution (run from project root):

**Behavioral Cloning (BC):**
```bash
tmux kill-session -t bc 2>/dev/null || true; tmux new -s bc bash -c 'source .env && python algorithms/offline/any_percent_bc_minari.py; exec bash'
```

**Advantage-Weighted Actor-Critic (AWAC):**
```bash
tmux kill-session -t awac 2>/dev/null || true; tmux new -s awac bash -c 'source .env && python algorithms/offline/awac_minari.py; exec bash'
```

**Online DQN via Papermill:**
```bash
tmux kill-session -t dqn 2>/dev/null || true; tmux new -s dqn bash -c 'source .env && papermill algorithms/online/dqn.ipynb algorithms/online/dqn_executed.ipynb; exec bash'
```

```zsh
tmux kill-session -t dqn 2>/dev/null || true; tmux new -s dqn zsh -c 'source .env && papermill algorithms/online/dqn.ipynb algorithms/online/dqn_executed.ipynb; exec zsh'
```

The DQN notebook `algorithms/online/dqn.ipynb` contains several preset configurations (e.g. Minigrid BabyAI, MiniGrid-Empty-5x5, CartPole). Edit the configuration cell at the top of the notebook to choose the desired environment and hyperparameters before running it with papermill.

2. Create a `.env` file in the project root:
```env
MLFLOW_HOST=localhost
MLFLOW_PORT=5001
```<br/>

* 📜 Single-file implementation
* 📈 Benchmarked Implementation for N algorithms
* 🖼 [Weights and Biases](https://wandb.ai/site) integration

----
* ⭐ If you're interested in __discrete control__, make sure to check out our new library — [Katakomba](https://github.com/tinkoff-ai/katakomba). It provides both discrete control algorithms augmented with recurrence and an offline RL benchmark for the NetHack Learning environment.
----


## Getting started

```bash
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL
pip install -r requirements/requirements_dev.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```


## Algorithms Implemented

| Algorithm                                                                                                                      | Variants Implemented                                                                                     | Wandb Report |
|--------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------| ----------- |
| **Offline and Offline-to-Online**                                                                                              |                                                                                                          |
| ✅ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)                     | [`offline/cql.py`](algorithms/offline/cql.py) <br /> [`finetune/cql.py`](algorithms/finetune/cql.py)     | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-CQL--VmlldzoyNzA2MTk5) <br /> <br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-CQL--Vmlldzo0NTQ3NTMz)
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)              | [`offline/awac.py`](algorithms/offline/awac.py) <br /> [`finetune/awac.py`](algorithms/finetune/awac.py) | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-AWAC--VmlldzoyNzA2MjE3) <br /> <br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-AWAC--VmlldzozODAyNzQz)
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)                        | [`offline/iql.py`](algorithms/offline/iql.py)  <br /> [`finetune/iql.py`](algorithms/finetune/iql.py)    | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-IQL--VmlldzoyNzA2MTkx) <br /> <br /> [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-IQL--VmlldzozNzE1MTEy)
| **Offline-to-Online only**                                                                                                     |                                                                                                          |
| ✅ [Supported Policy Optimization for Offline Reinforcement Learning <br>(SPOT)](https://arxiv.org/abs/2202.06239)              | [`finetune/spot.py`](algorithms/finetune/spot.py)                                                        | [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-SPOT--VmlldzozODk5MTgx)
| ✅ [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning <br>(Cal-QL)](https://arxiv.org/abs/2303.05479) | [`finetune/cal_ql.py`](algorithms/finetune/cal_ql.py)                                                    | [`Offline-to-online`](https://wandb.ai/tlab/CORL/reports/-Offline-to-Online-Cal-QL--Vmlldzo0NTQ3NDk5)
| **Offline only**                                                                                                               |                                                                                                          |
| ✅ Behavioral Cloning <br>(BC)                                                                                                  | [`offline/any_percent_bc.py`](algorithms/offline/any_percent_bc.py)                                      |  [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC--VmlldzoyNzA2MjE1)
| ✅ Behavioral Cloning-10% <br>(BC-10%)                                                                                          | [`offline/any_percent_bc.py`](algorithms/offline/any_percent_bc.py)                                      |  [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-BC-10---VmlldzoyNzEwMjcx)
| ✅ [A Minimalist Approach to Offline Reinforcement Learning <br>(TD3+BC)](https://arxiv.org/abs/2106.06860)                     | [`offline/td3_bc.py`](algorithms/offline/td3_bc.py)                                                      | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-TD3-BC--VmlldzoyNzA2MjA0)
| ✅ [Decision Transformer: Reinforcement Learning via Sequence Modeling <br>(DT)](https://arxiv.org/abs/2106.01345)              | [`offline/dt.py`](algorithms/offline/dt.py)                                                              | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-Decision-Transformer--VmlldzoyNzA2MTk3)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(SAC-N)](https://arxiv.org/abs/2110.01548) | [`offline/sac_n.py`](algorithms/offline/sac_n.py)                                                        | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-SAC-N--VmlldzoyNzA1NTY1)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(EDAC)](https://arxiv.org/abs/2110.01548)  | [`offline/edac.py`](algorithms/offline/edac.py)                                                          | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-EDAC--VmlldzoyNzA5ODUw)
| ✅ [Revisiting the Minimalist Approach to Offline Reinforcement Learning <br>(ReBRAC)](https://arxiv.org/abs/2305.09836)        | [`offline/rebrac.py`](algorithms/offline/rebrac.py)                                                      | [`Offline`](https://wandb.ai/tlab/CORL/reports/-Offline-ReBRAC--Vmlldzo0ODkzOTQ2)
| ✅ [Q-Ensemble for Offline RL: Don't Scale the Ensemble, Scale the Batch Size <br>(LB-SAC)](https://arxiv.org/abs/2211.11092)   | [`offline/lb_sac.py`](algorithms/offline/lb_sac.py)                                                      | [`Offline Gym-MuJoCo`](https://wandb.ai/tlab/CORL/reports/LB-SAC-D4RL-Results--VmlldzozNjIxMDY1)


## D4RL Benchmarks
You can check the links above for learning curves and details. Here, we report reproduced **final** and **best** scores. Note that they differ by a significant margin, and some papers may use different approaches, not making it always explicit which reporting methodology they chose. If you want to re-collect our results in a more structured/nuanced manner, see [`results`](results).

### Offline
#### Last Scores
##### Gym-MuJoCo

| **Task-Name**|BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|------------------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2|42.40 ± 0.19|42.46 ± 0.70|48.10 ± 0.18|49.46 ± 0.62|47.04 ± 0.22|48.31 ± 0.22|64.04 ± 0.68|68.20 ± 1.28|67.70 ± 1.04|42.20 ± 0.26|
|halfcheetah-medium-replay-v2|35.66 ± 2.33|23.59 ± 6.95|44.84 ± 0.59|44.70 ± 0.69|45.04 ± 0.27|44.46 ± 0.22|51.18 ± 0.31|60.70 ± 1.01|62.06 ± 1.10|38.91 ± 0.50|
|halfcheetah-medium-expert-v2|55.95 ± 7.35|90.10 ± 2.45|90.78 ± 6.04|93.62 ± 0.41|95.63 ± 0.42|94.74 ± 0.52|103.80 ± 2.95|98.96 ± 9.31|104.76 ± 0.64|91.55 ± 0.95|
|hopper-medium-v2|53.51 ± 1.76|55.48 ± 7.30|60.37 ± 3.49|74.45 ± 9.14|59.08 ± 3.77|67.53 ± 3.78|102.29 ± 0.17|40.82 ± 9.91|101.70 ± 0.28|65.10 ± 1.61|
|hopper-medium-replay-v2|29.81 ± 2.07|70.42 ± 8.66|64.42 ± 21.52|96.39 ± 5.28|95.11 ± 5.27|97.43 ± 6.39|94.98 ± 6.53|100.33 ± 0.78|99.66 ± 0.81|81.77 ± 6.87|
|hopper-medium-expert-v2|52.30 ± 4.01|111.16 ± 1.03|101.17 ± 9.07|52.73 ± 37.47|99.26 ± 10.91|107.42 ± 7.80|109.45 ± 2.34|101.31 ± 11.63|105.19 ± 10.08|110.44 ± 0.33|
|walker2d-medium-v2|63.23 ± 16.24|67.34 ± 5.17|82.71 ± 4.78|66.53 ± 26.04|80.75 ± 3.28|80.91 ± 3.17|85.82 ± 0.77|87.47 ± 0.66|93.36 ± 1.38|67.63 ± 2.54|
|walker2d-medium-replay-v2|21.80 ± 10.15|54.35 ± 6.34|85.62 ± 4.01|82.20 ± 1.05|73.09 ± 13.22|82.15 ± 3.03|84.25 ± 2.25|78.99 ± 0.50|87.10 ± 2.78|59.86 ± 2.73|
|walker2d-medium-expert-v2|98.96 ± 15.98|108.70 ± 0.25|110.03 ± 0.36|49.41 ± 38.16|109.56 ± 0.39|111.72 ± 0.86|111.86 ± 0.43|114.93 ± 0.41|114.75 ± 0.74|107.11 ± 0.96|
|                              |            |        |        |     |     |      |       |      |    |  |
| **locomotion average**       |50.40|69.29|76.45|67.72|78.28|81.63|89.74|83.52|92.92|73.84|

##### Maze2d
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
| maze2d-umaze-v1    |0.36 ± 8.69|12.18 ± 4.29|29.41 ± 12.31|82.67 ± 28.30|-8.90 ± 6.11|42.11 ± 0.58|106.87 ± 22.16|130.59 ± 16.52|95.26 ± 6.39|18.08 ± 25.42|
| maze2d-medium-v1   |0.79 ± 3.25|14.25 ± 2.33|59.45 ± 36.25|52.88 ± 55.12|86.11 ± 9.68|34.85 ± 2.72|105.11 ± 31.67|88.61 ± 18.72|57.04 ± 3.45|31.71 ± 26.33|
| maze2d-large-v1    |2.26 ± 4.39|11.32 ± 5.10|97.10 ± 25.41|209.13 ± 8.19|23.75 ± 36.70|61.72 ± 3.50|78.33 ± 61.77|204.76 ± 1.19|95.60 ± 22.92|35.66 ± 28.20|
|                    |            |        |        |     |     |      |       |      |    |  |
| **maze2d average** |1.13|12.58|61.99|114.89|33.65|46.23|96.77|141.32|82.64|28.48|

##### Antmaze
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v2|55.25 ± 4.15|65.75 ± 5.26|70.75 ± 39.18|57.75 ± 10.28|92.75 ± 1.92|77.00 ± 5.52|97.75 ± 1.48|0.00 ± 0.00|0.00 ± 0.00|57.00 ± 9.82|
|antmaze-umaze-diverse-v2|47.25 ± 4.09|44.00 ± 1.00|44.75 ± 11.61|58.00 ± 7.68|37.25 ± 3.70|54.25 ± 5.54|83.50 ± 7.02|0.00 ± 0.00|0.00 ± 0.00|51.75 ± 0.43|
|antmaze-medium-play-v2|0.00 ± 0.00|2.00 ± 0.71|0.25 ± 0.43|0.00 ± 0.00|65.75 ± 11.61|65.75 ± 11.71|89.50 ± 3.35|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|antmaze-medium-diverse-v2|0.75 ± 0.83|5.75 ± 9.39|0.25 ± 0.43|0.00 ± 0.00|67.25 ± 3.56|73.75 ± 5.45|83.50 ± 8.20|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|antmaze-large-play-v2|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|20.75 ± 7.26|42.00 ± 4.53|52.25 ± 29.01|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|antmaze-large-diverse-v2|0.00 ± 0.00|0.75 ± 0.83|0.00 ± 0.00|0.00 ± 0.00|20.50 ± 13.24|30.25 ± 3.63|64.00 ± 5.43|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|                    |            |        |        |     |     |      |       |      |    |  |
| **antmaze average**           | 17.21|19.71|19.33|19.29|50.71|57.17|78.42|0.00|0.00|18.12|

##### Adroit
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|pen-human-v1|71.03 ± 6.26|26.99 ± 9.60|-3.88 ± 0.21|81.12 ± 13.47|13.71 ± 16.98|78.49 ± 8.21|103.16 ± 8.49|6.86 ± 5.93|5.07 ± 6.16|67.68 ± 5.48|
|pen-cloned-v1|51.92 ± 15.15|46.67 ± 14.25|5.13 ± 5.28|89.56 ± 15.57|1.04 ± 6.62|83.42 ± 8.19|102.79 ± 7.84|31.35 ± 2.14|12.02 ± 1.75|64.43 ± 1.43|
|pen-expert-v1|109.65 ± 7.28|114.96 ± 2.96|122.53 ± 21.27|160.37 ± 1.21|-1.41 ± 2.34|128.05 ± 9.21|152.16 ± 6.33|87.11 ± 48.95|-1.55 ± 0.81|116.38 ± 1.27|
|door-human-v1|2.34 ± 4.00|-0.13 ± 0.07|-0.33 ± 0.01|4.60 ± 1.90|5.53 ± 1.31|3.26 ± 1.83|-0.10 ± 0.01|-0.38 ± 0.00|-0.12 ± 0.13|4.44 ± 0.87|
|door-cloned-v1|-0.09 ± 0.03|0.29 ± 0.59|-0.34 ± 0.01|0.93 ± 1.66|-0.33 ± 0.01|3.07 ± 1.75|0.06 ± 0.05|-0.33 ± 0.00|2.66 ± 2.31|7.64 ± 3.26|
|door-expert-v1|105.35 ± 0.09|104.04 ± 1.46|-0.33 ± 0.01|104.85 ± 0.24|-0.32 ± 0.02|106.65 ± 0.25|106.37 ± 0.29|-0.33 ± 0.00|106.29 ± 1.73|104.87 ± 0.39|
|hammer-human-v1|3.03 ± 3.39|-0.19 ± 0.02|1.02 ± 0.24|3.37 ± 1.93|0.14 ± 0.11|1.79 ± 0.80|0.24 ± 0.24|0.24 ± 0.00|0.28 ± 0.18|1.28 ± 0.15|
|hammer-cloned-v1|0.55 ± 0.16|0.12 ± 0.08|0.25 ± 0.01|0.21 ± 0.24|0.30 ± 0.01|1.50 ± 0.69|5.00 ± 3.75|0.14 ± 0.09|0.19 ± 0.07|1.82 ± 0.55|
|hammer-expert-v1|126.78 ± 0.64|121.75 ± 7.67|3.11 ± 0.03|127.06 ± 0.29|0.26 ± 0.01|128.68 ± 0.33|133.62 ± 0.27|25.13 ± 43.25|28.52 ± 49.00|117.45 ± 6.65|
|relocate-human-v1|0.04 ± 0.03|-0.14 ± 0.08|-0.29 ± 0.01|0.05 ± 0.03|0.06 ± 0.03|0.12 ± 0.04|0.16 ± 0.30|-0.31 ± 0.01|-0.17 ± 0.17|0.05 ± 0.01|
|relocate-cloned-v1|-0.06 ± 0.01|-0.00 ± 0.02|-0.30 ± 0.01|-0.04 ± 0.04|-0.29 ± 0.01|0.04 ± 0.01|1.66 ± 2.59|-0.01 ± 0.10|0.17 ± 0.35|0.16 ± 0.09|
|relocate-expert-v1|107.58 ± 1.20|97.90 ± 5.21|-1.73 ± 0.96|108.87 ± 0.85|-0.30 ± 0.02|106.11 ± 4.02|107.52 ± 2.28|-0.36 ± 0.00|71.94 ± 18.37|104.28 ± 0.42|
|                    |            |        |        |     |     |      |       |      |    |  |
| **adroit average**        | 48.18|42.69|10.40|56.75|1.53|53.43|59.39|12.43|18.78|49.21|

#### Best Scores
##### Gym-MuJoCo
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2|43.60 ± 0.14|43.90 ± 0.13|48.93 ± 0.11|50.06 ± 0.50|47.62 ± 0.03|48.84 ± 0.07|65.62 ± 0.46|72.21 ± 0.31|69.72 ± 0.92|42.73 ± 0.10|
|halfcheetah-medium-replay-v2|40.52 ± 0.19|42.27 ± 0.46|45.84 ± 0.26|46.35 ± 0.29|46.43 ± 0.19|45.35 ± 0.08|52.22 ± 0.31|67.29 ± 0.34|66.55 ± 1.05|40.31 ± 0.28|
|halfcheetah-medium-expert-v2|79.69 ± 3.10|94.11 ± 0.22|96.59 ± 0.87|96.11 ± 0.37|97.04 ± 0.17|95.38 ± 0.17|108.89 ± 1.20|111.73 ± 0.47|110.62 ± 1.04|93.40 ± 0.21|
|hopper-medium-v2|69.04 ± 2.90|73.84 ± 0.37|70.44 ± 1.18|97.90 ± 0.56|70.80 ± 1.98|80.46 ± 3.09|103.19 ± 0.16|101.79 ± 0.20|103.26 ± 0.14|69.42 ± 3.64|
|hopper-medium-replay-v2|68.88 ± 10.33|90.57 ± 2.07|98.12 ± 1.16|100.91 ± 1.50|101.63 ± 0.55|102.69 ± 0.96|102.57 ± 0.45|103.83 ± 0.53|103.28 ± 0.49|88.74 ± 3.02|
|hopper-medium-expert-v2|90.63 ± 10.98|113.13 ± 0.16|113.22 ± 0.43|103.82 ± 12.81|112.84 ± 0.66|113.18 ± 0.38|113.16 ± 0.43|111.24 ± 0.15|111.80 ± 0.11|111.18 ± 0.21|
|walker2d-medium-v2|80.64 ± 0.91|82.05 ± 0.93|86.91 ± 0.28|83.37 ± 2.82|84.77 ± 0.20|87.58 ± 0.48|87.79 ± 0.19|90.17 ± 0.54|95.78 ± 1.07|74.70 ± 0.56|
|walker2d-medium-replay-v2|48.41 ± 7.61|76.09 ± 0.40|91.17 ± 0.72|86.51 ± 1.15|89.39 ± 0.88|89.94 ± 0.93|91.11 ± 0.63|85.18 ± 1.63|89.69 ± 1.39|68.22 ± 1.20|
|walker2d-medium-expert-v2|109.95 ± 0.62|109.90 ± 0.09|112.21 ± 0.06|108.28 ± 9.45|111.63 ± 0.38|113.06 ± 0.53|112.49 ± 0.18|116.93 ± 0.42|116.52 ± 0.75|108.71 ± 0.34|
|                    |            |        |        |     |     |      |       |      |    |  |
| **locomotion average**       |    70.15|80.65|84.83|85.92|84.68|86.28|93.00|95.60|96.36|77.49|


##### Maze2d
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1|16.09 ± 0.87|22.49 ± 1.52|99.33 ± 16.16|136.61 ± 11.65|92.05 ± 13.66|50.92 ± 4.23|162.28 ± 1.79|153.12 ± 6.49|149.88 ± 1.97|63.83 ± 17.35|
|maze2d-medium-v1|19.16 ± 1.24|27.64 ± 1.87|150.93 ± 3.89|131.50 ± 25.38|128.66 ± 5.44|122.69 ± 30.00|150.12 ± 4.48|93.80 ± 14.66|154.41 ± 1.58|68.14 ± 12.25|
|maze2d-large-v1|20.75 ± 6.66|41.83 ± 3.64|197.64 ± 5.26|227.93 ± 1.90|157.51 ± 7.32|162.25 ± 44.18|197.55 ± 5.82|207.51 ± 0.96|182.52 ± 2.68|50.25 ± 19.34|
|                    |            |        |        |     |     |      |       |      |    |  |
| **maze2d average**           | 18.67|30.65|149.30|165.35|126.07|111.95|169.98|151.48|162.27|60.74|

##### Antmaze
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v2|68.50 ± 2.29|77.50 ± 1.50|98.50 ± 0.87|78.75 ± 6.76|94.75 ± 0.83|84.00 ± 4.06|100.00 ± 0.00|0.00 ± 0.00|42.50 ± 28.61|64.50 ± 2.06|
|antmaze-umaze-diverse-v2|64.75 ± 4.32|63.50 ± 2.18|71.25 ± 5.76|88.25 ± 2.17|53.75 ± 2.05|79.50 ± 3.35|96.75 ± 2.28|0.00 ± 0.00|0.00 ± 0.00|60.50 ± 2.29|
|antmaze-medium-play-v2|4.50 ± 1.12|6.25 ± 2.38|3.75 ± 1.30|27.50 ± 9.39|80.50 ± 3.35|78.50 ± 3.84|93.50 ± 2.60|0.00 ± 0.00|0.00 ± 0.00|0.75 ± 0.43|
|antmaze-medium-diverse-v2|4.75 ± 1.09|16.50 ± 5.59|5.50 ± 1.50|33.25 ± 16.81|71.00 ± 4.53|83.50 ± 1.80|91.75 ± 2.05|0.00 ± 0.00|0.00 ± 0.00|0.50 ± 0.50|
|antmaze-large-play-v2|0.50 ± 0.50|13.50 ± 9.76|1.25 ± 0.43|1.00 ± 0.71|34.75 ± 5.85|53.50 ± 2.50|68.75 ± 13.90|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|antmaze-large-diverse-v2|0.75 ± 0.43|6.25 ± 1.79|0.25 ± 0.43|0.50 ± 0.50|36.25 ± 3.34|53.00 ± 3.00|69.50 ± 7.26|0.00 ± 0.00|0.00 ± 0.00|0.00 ± 0.00|
|                    |            |        |        |     |     |      |       |      |    |  |
| **antmaze average**           |23.96|30.58|30.08|38.21|61.83|72.00|86.71|0.00|7.08|21.04|

##### Adroit
| **Task-Name**      |BC|10% BC|TD3+BC|AWAC|CQL|IQL|ReBRAC|SAC-N|EDAC|DT|
|--------------------|------------|--------|--------|--------|-----|-----|------|-------|------|----|
|pen-human-v1|99.69 ± 7.45|59.89 ± 8.03|9.95 ± 8.19|121.05 ± 5.47|58.91 ± 1.81|106.15 ± 10.28|127.28 ± 3.22|56.48 ± 7.17|35.84 ± 10.57|77.83 ± 2.30|
|pen-cloned-v1|99.14 ± 12.27|83.62 ± 11.75|52.66 ± 6.33|129.66 ± 1.27|14.74 ± 2.31|114.05 ± 4.78|128.64 ± 7.15|52.69 ± 5.30|26.90 ± 7.85|71.17 ± 2.70|
|pen-expert-v1|128.77 ± 5.88|134.36 ± 3.16|142.83 ± 7.72|162.69 ± 0.23|14.86 ± 4.07|140.01 ± 6.36|157.62 ± 0.26|116.43 ± 40.26|36.04 ± 4.60|119.49 ± 2.31|
|door-human-v1|9.41 ± 4.55|7.00 ± 6.77|-0.11 ± 0.06|19.28 ± 1.46|13.28 ± 2.77|13.52 ± 1.22|0.27 ± 0.43|-0.10 ± 0.06|2.51 ± 2.26|7.36 ± 1.24|
|door-cloned-v1|3.40 ± 0.95|10.37 ± 4.09|-0.20 ± 0.11|12.61 ± 0.60|-0.08 ± 0.13|9.02 ± 1.47|7.73 ± 6.80|-0.21 ± 0.10|20.36 ± 1.11|11.18 ± 0.96|
|door-expert-v1|105.84 ± 0.23|105.92 ± 0.24|4.49 ± 7.39|106.77 ± 0.24|59.47 ± 25.04|107.29 ± 0.37|106.78 ± 0.04|0.05 ± 0.02|109.22 ± 0.24|105.49 ± 0.09|
|hammer-human-v1|12.61 ± 4.87|6.23 ± 4.79|2.38 ± 0.14|22.03 ± 8.13|0.30 ± 0.05|6.86 ± 2.38|1.18 ± 0.15|0.25 ± 0.00|3.49 ± 2.17|1.68 ± 0.11|
|hammer-cloned-v1|8.90 ± 4.04|8.72 ± 3.28|0.96 ± 0.30|14.67 ± 1.94|0.32 ± 0.03|11.63 ± 1.70|48.16 ± 6.20|12.67 ± 15.02|0.27 ± 0.01|2.74 ± 0.22|
|hammer-expert-v1|127.89 ± 0.57|128.15 ± 0.66|33.31 ± 47.65|129.66 ± 0.33|0.93 ± 1.12|129.76 ± 0.37|134.74 ± 0.30|91.74 ± 47.77|69.44 ± 47.00|127.39 ± 0.10|
|relocate-human-v1|0.59 ± 0.27|0.16 ± 0.14|-0.29 ± 0.01|2.09 ± 0.76|1.03 ± 0.20|1.22 ± 0.28|3.70 ± 2.34|-0.18 ± 0.14|0.05 ± 0.02|0.08 ± 0.02|
|relocate-cloned-v1|0.45 ± 0.31|0.74 ± 0.45|-0.02 ± 0.04|0.94 ± 0.68|-0.07 ± 0.02|1.78 ± 0.70|9.25 ± 2.56|0.10 ± 0.04|4.11 ± 1.39|0.34 ± 0.09|
|relocate-expert-v1|110.31 ± 0.36|109.77 ± 0.60|0.23 ± 0.27|111.56 ± 0.17|0.03 ± 0.10|110.12 ± 0.82|111.14 ± 0.23|-0.07 ± 0.08|98.32 ± 3.75|106.49 ± 0.30|
|                    |            |        |        |     |     |      |       |      |    |  |
| **adroit average** | 58.92|54.58|20.51|69.42|13.65|62.62|69.71|27.49|33.88|52.60|

### Offline-to-Online
#### Scores
| **Task-Name**             |AWAC|CQL|IQL|SPOT|Cal-QL|
|---------------------------|------------|--------|--------|-----|-----|
|antmaze-umaze-v2|52.75 ± 8.67 →  98.75 ± 1.09|94.00 ± 1.58 →  99.50 ± 0.87|77.00 ± 0.71 →  96.50 ± 1.12|91.00 ± 2.55 →  99.50 ± 0.50|76.75 ± 7.53 →  99.75 ± 0.43|
|antmaze-umaze-diverse-v2|56.00 ± 2.74 →  0.00 ± 0.00|9.50 ± 9.91 →  99.00 ± 1.22|59.50 ± 9.55 →  63.75 ± 25.02|36.25 ± 2.17 →  95.00 ± 3.67|32.00 ± 27.79 →  98.50 ± 1.12|
|antmaze-medium-play-v2|0.00 ± 0.00 →  0.00 ± 0.00|59.00 ± 11.18 →  97.75 ± 1.30|71.75 ± 2.95 →  89.75 ± 1.09|67.25 ± 10.47 →  97.25 ± 1.30|71.75 ± 3.27 →  98.75 ± 1.64|
|antmaze-medium-diverse-v2|0.00 ± 0.00 →  0.00 ± 0.00|63.50 ± 6.84 →  97.25 ± 1.92|64.25 ± 1.92 →  92.25 ± 2.86|73.75 ± 7.29 →  94.50 ± 1.66|62.00 ± 4.30 →  98.25 ± 1.48|
|antmaze-large-play-v2|0.00 ± 0.00 →  0.00 ± 0.00|28.75 ± 7.76 →  88.25 ± 2.28|38.50 ± 8.73 →  64.50 ± 17.04|31.50 ± 12.58 →  87.00 ± 3.24|31.75 ± 8.87 →  97.25 ± 1.79|
|antmaze-large-diverse-v2|0.00 ± 0.00 →  0.00 ± 0.00|35.50 ± 3.64 →  91.75 ± 3.96|26.75 ± 3.77 →  64.25 ± 4.15|17.50 ± 7.26 →  81.00 ± 14.14|44.00 ± 8.69 →  91.50 ± 3.91|
|                           |            |        |        |     |     |      |       |      |    |
| **antmaze average**       |18.12 →  16.46|48.38 →  95.58|56.29 →  78.50|52.88 →  92.38|53.04 →  97.33|
|                           |            |        |        |     |     |      |       |      |    |
|pen-cloned-v1|88.66 ± 15.10 →  86.82 ± 11.12|-2.76 ± 0.08 →  -1.28 ± 2.16|84.19 ± 3.96 →  102.02 ± 20.75|6.19 ± 5.21 →  43.63 ± 20.09|-2.66 ± 0.04 →  -2.68 ± 0.12|
|door-cloned-v1|0.93 ± 1.66 →  0.01 ± 0.00|-0.33 ± 0.01 →  -0.33 ± 0.01|1.19 ± 0.93 →  20.34 ± 9.32|-0.21 ± 0.14 →  0.02 ± 0.31|-0.33 ± 0.01 →  -0.33 ± 0.01|
|hammer-cloned-v1|1.80 ± 3.01 →  0.24 ± 0.04|0.56 ± 0.55 →  2.85 ± 4.81|1.35 ± 0.32 →  57.27 ± 28.49|3.97 ± 6.39 →  3.73 ± 4.99|0.25 ± 0.04 →  0.17 ± 0.17|
|relocate-cloned-v1|-0.04 ± 0.04 →  -0.04 ± 0.01|-0.33 ± 0.01 →  -0.33 ± 0.01|0.04 ± 0.04 →  0.32 ± 0.38|-0.24 ± 0.01 →  -0.15 ± 0.05|-0.31 ± 0.05 →  -0.31 ± 0.04|
|                           |            |        |        |     |     |      |       |      |    |
| **adroit average**        |22.84 →  21.76|-0.72 →  0.22|21.69 →  44.99|2.43 →  11.81|-0.76 →  -0.79|

#### Regrets
| **Task-Name**             |AWAC|CQL|IQL|SPOT|Cal-QL|
|---------------------------|------------|--------|--------|-----|-----|
|antmaze-umaze-v2|0.04 ± 0.01|0.02 ± 0.00|0.07 ± 0.00|0.02 ± 0.00|0.01 ± 0.00|
|antmaze-umaze-diverse-v2|0.88 ± 0.01|0.09 ± 0.01|0.43 ± 0.11|0.22 ± 0.07|0.05 ± 0.01|
|antmaze-medium-play-v2|1.00 ± 0.00|0.08 ± 0.01|0.09 ± 0.01|0.06 ± 0.00|0.04 ± 0.01|
|antmaze-medium-diverse-v2|1.00 ± 0.00|0.08 ± 0.00|0.10 ± 0.01|0.05 ± 0.01|0.04 ± 0.01|
|antmaze-large-play-v2|1.00 ± 0.00|0.21 ± 0.02|0.34 ± 0.05|0.29 ± 0.07|0.13 ± 0.02|
|antmaze-large-diverse-v2|1.00 ± 0.00|0.21 ± 0.03|0.41 ± 0.03|0.23 ± 0.08|0.13 ± 0.02|
|                           |            |        |        |     |     |      |       |      |    |
| **antmaze average**       |0.82|0.11|0.24|0.15|0.07|
|                           |            |        |        |     |     |      |       |      |    |
|pen-cloned-v1|0.46 ± 0.02|0.97 ± 0.00|0.37 ± 0.01|0.58 ± 0.02|0.98 ± 0.01|
|door-cloned-v1|1.00 ± 0.00|1.00 ± 0.00|0.83 ± 0.03|0.99 ± 0.01|1.00 ± 0.00|
|hammer-cloned-v1|1.00 ± 0.00|1.00 ± 0.00|0.65 ± 0.10|0.98 ± 0.01|1.00 ± 0.00|
|relocate-cloned-v1|1.00 ± 0.00|1.00 ± 0.00|1.00 ± 0.00|1.00 ± 0.00|1.00 ± 0.00|
|                           |            |        |        |     |     |      |       |      |    |
| **adroit average**        |0.86|0.99|0.71|0.89|0.99|

## Citing CORL

If you use CORL in your work, please use the following bibtex
```bibtex
@inproceedings{
tarasov2022corl,
  title={{CORL}: Research-oriented Deep Offline Reinforcement Learning Library},
  author={Denis Tarasov and Alexander Nikulin and Dmitry Akimov and Vladislav Kurenkov and Sergey Kolesnikov},
  booktitle={3rd Offline RL Workshop: Offline RL as a ''Launchpad''},
  year={2022},
  url={https://openreview.net/forum?id=SyAS49bBcv}
}
```
