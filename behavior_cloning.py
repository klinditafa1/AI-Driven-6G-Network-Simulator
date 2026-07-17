"""
Warm-start a PPO policy by training it (via supervised cross-entropy) to
imitate Max-SINR's actions on the collected expert dataset. The resulting
weights are saved and then fine-tuned with RL in finetune_from_bc.py.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from stable_baselines3 import PPO
from envs.network_env import NetworkSchedulingEnv

EXPERT_DATA_PATH = "results/trained_models/expert_data.npz"
BC_MODEL_PATH = "results/trained_models/ppo_bc_warmstart"
EPOCHS = 40
BATCH_SIZE = 256
LR = 3e-4


def main():
    data = np.load(EXPERT_DATA_PATH)
    obs_all, act_all = data["obs"], data["actions"]
    n = len(obs_all)
    print(f"Loaded {n} expert transitions")

    env = NetworkSchedulingEnv(n_users_per_cell=8, n_rb=3, episode_len=200, seed=0)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=256,
                learning_rate=1e-4, gamma=0.97, ent_coef=0.01, clip_range=0.2,
                n_epochs=10, seed=0)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    obs_t = torch.as_tensor(obs_all, dtype=torch.float32)
    act_t = torch.as_tensor(act_all, dtype=torch.long)

    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            obs_b, act_b = obs_t[idx], act_t[idx]

            dist = policy.get_distribution(obs_b)
            # MultiCategoricalDistribution wraps a list of per-dimension Categoricals
            log_prob = torch.stack(
                [d.log_prob(act_b[:, k]) for k, d in enumerate(dist.distribution)], dim=1
            ).sum(dim=1)
            loss = -log_prob.mean()

            preds = torch.stack([d.probs.argmax(dim=1) for d in dist.distribution], dim=1)
            acc = (preds == act_b).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        print(f"epoch {epoch+1}/{EPOCHS}: loss={total_loss/n_batches:.4f} "
              f"per-dim action accuracy={total_acc/n_batches:.3f}")

    model.save(BC_MODEL_PATH)
    print(f"Saved behavior-cloned model to {BC_MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
