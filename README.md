# AI-Driven 6G Network Simulator

A simulator of a small cellular network — base stations, users, interference,
traffic — built from scratch in Python, paired with a reinforcement learning
agent that learns how to schedule network resources.

The idea: model how a 5G/6G network actually behaves (signal strength,
interference between cells, users competing for bandwidth), then see if an
RL agent can learn to schedule resources better than the classic algorithms
used in real networks today (Round Robin, Max-SINR, Proportional Fair).

## What it does

- Places base stations and users on a map and simulates signal strength,
  interference, and data throughput between them.
- Runs three standard scheduling algorithms as a baseline for comparison.
- Trains an RL agent (PPO) to do the scheduling instead, and compares it
  against those baselines.
- When the agent trained from scratch didn't beat the best baseline, I tried
  warm-starting it by having it imitate the baseline first, then fine-tuning
  with RL — which helped, though it still hasn't fully closed the gap.

## Results, honestly

The RL agent doesn't beat the best baseline (Max-SINR) yet. It gets closer
with the imitation-learning warm start than training from scratch, but there's
still a gap. The plots and a full writeup of what I tried and what I'd try
next are in the `results/` folder.

## Running it

```bash
pip install -r requirements.txt

python3 evaluate_baselines.py     # simulate the network, compare baseline schedulers
python3 train_increment.py        # train an RL agent (rerun to keep training)
python3 collect_expert_data.py    # collect data for the imitation warm-start
python3 behavior_cloning.py
python3 finetune_from_bc.py       # rerun to keep fine-tuning
python3 evaluate_agent.py         # final comparison of everything
```
