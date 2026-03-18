# Econ-RL

An exploration of Deep Reinforcement Learning (DRL) from an economist's perspective.

This repository serves as a learning journey, starting with basic methods and standard libraries applied to conventional problems (e.g., CartPole), and eventually moving towards applications with an economic flavor — including a custom Bewley-style savings environment where an agent learns to optimize consumption under income uncertainty.

## Structure

- `projects/cartpole/`: REINFORCE and DQN baselines for CartPole-v1
- `projects/income_fluctuation/`: Custom income fluctuation environment with training and policy validation
- `projects/lunar_lander/`: LunarLander experiment

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Econ-RL
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run scripts from the repo root:

```bash
python projects/cartpole/train.py --episodes 1000
python projects/income_fluctuation/train.py --episodes 1000 --batch_size 10
```
