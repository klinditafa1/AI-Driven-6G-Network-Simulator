"""
Collect (observation, expert_action) pairs by running the Max-SINR baseline
policy in the RL environment. Used to warm-start the PPO policy via behavior
cloning before RL fine-tuning, since from-scratch PPO plateaued below
Max-SINR's performance (see README).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from envs.network_env import NetworkSchedulingEnv
from baselines.max_sinr import MaxSINRScheduler

N_EPISODES = 300
EPISODE_LEN = 200
OUT_PATH = "results/trained_models/expert_data.npz"


def collect():
    scheduler = MaxSINRScheduler()
    all_obs, all_actions = [], []

    for ep in range(N_EPISODES):
        env = NetworkSchedulingEnv(n_users_per_cell=8, n_rb=3, episode_len=EPISODE_LEN, seed=5000 + ep)
        obs, info = env.reset(seed=5000 + ep)
        done = False
        while not done:
            bs = env._bs_by_id[0]
            assignment = scheduler(bs, env.controlled_users, env.sim)
            action = []
            for rb_idx in range(env.n_rb):
                u = assignment[rb_idx] if rb_idx < len(assignment) else None
                action.append(env.controlled_users.index(u) if u is not None else 0)
            action = np.array(action)

            all_obs.append(obs.copy())
            all_actions.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if (ep + 1) % 50 == 0:
            print(f"collected {ep + 1}/{N_EPISODES} episodes, {len(all_obs)} transitions so far")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int64)
    np.savez(OUT_PATH, obs=all_obs, actions=all_actions)
    print(f"Saved {len(all_obs)} (obs, action) pairs to {OUT_PATH}")


if __name__ == "__main__":
    collect()
