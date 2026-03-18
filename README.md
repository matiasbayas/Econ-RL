# Econ-RL

Reinforcement learning experiments for economic models and standard control tasks.

This repository combines baseline RL exercises with a custom income fluctuation environment. The main economics example is a Bewley-style savings problem: an agent receives stochastic income, chooses a savings rate each period, and learns a policy that trades off current consumption against future assets.

## Projects

- `projects/cartpole/`: REINFORCE and DQN baselines for `CartPole-v1`, plus comparison and visualization scripts.
- `projects/income_fluctuation/`: Custom income fluctuation environment, REINFORCE training, policy validation, and variance checks.
- `projects/lunar_lander/`: Small LunarLander environment experiment.

## Installation

```bash
git clone <repository_url>
cd Econ-RL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train CartPole with REINFORCE:

```bash
python projects/cartpole/train.py --episodes 1000
```

Train CartPole with DQN:

```bash
python projects/cartpole/dqn_train.py --episodes 500
```

Compare the latest CartPole runs:

```bash
python projects/cartpole/compare_results.py
python projects/cartpole/compare_reinforce_variants.py
```

Train the income fluctuation agent:

```bash
python projects/income_fluctuation/train.py --episodes 1000 --batch_size 10
```

Validate a trained income fluctuation policy:

```bash
python projects/income_fluctuation/validate_agent.py projects/income_fluctuation/results/run_<timestamp>
```

## Repository Structure

```text
projects/
  cartpole/
  income_fluctuation/
  lunar_lander/
requirements.txt
README.md
```

## What This Repo Demonstrates

- Policy-gradient and value-based RL implementations in PyTorch.
- A custom RL environment for an economic savings problem with stochastic income.
- Post-training validation that converts learned policies into consumption and asset policy functions.
