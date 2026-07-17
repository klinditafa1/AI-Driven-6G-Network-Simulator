import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.network_env import NetworkSchedulingEnv
from baselines.round_robin import RoundRobinScheduler
from baselines.max_sinr import MaxSINRScheduler
from baselines.proportional_fair import ProportionalFairScheduler

N_EVAL_EPISODES = 10
EPISODE_LEN = 200


def eval_policy_on_env(policy_fn, seed_base=1000):
    """policy_fn(env, obs) -> action. Runs N_EVAL_EPISODES fresh episodes."""
    all_throughput, all_fairness, all_latency = [], [], []
    for ep in range(N_EVAL_EPISODES):
        env = NetworkSchedulingEnv(n_users_per_cell=8, n_rb=3, episode_len=EPISODE_LEN,
                                    seed=seed_base + ep)
        obs, info = env.reset(seed=seed_base + ep)
        ep_tp, ep_fair, ep_lat = [], [], []
        done = False
        while not done:
            action = policy_fn(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_tp.append(info["controlled_throughput_mbps"])
            ep_fair.append(info["controlled_fairness"])
            ep_lat.append(info["controlled_latency"])
            done = terminated or truncated
        all_throughput.append(np.mean(ep_tp))
        all_fairness.append(np.mean(ep_fair))
        all_latency.append(np.mean(ep_lat))
    return {
        "throughput_mbps": (np.mean(all_throughput), np.std(all_throughput)),
        "fairness": (np.mean(all_fairness), np.std(all_fairness)),
        "latency_slots": (np.mean(all_latency), np.std(all_latency)),
    }


def make_baseline_policy(scheduler_instance):
    """Wraps a baseline scheduler (designed for NetworkSim) into an env-action policy
    by having it choose users directly for the controlled cell's resource blocks."""
    def policy_fn(env, obs):
        bs = env._bs_by_id[0]
        assignment = scheduler_instance(bs, env.controlled_users, env.sim)
        action = []
        for rb_idx in range(env.n_rb):
            u = assignment[rb_idx] if rb_idx < len(assignment) else None
            if u is None:
                action.append(0)  # no-op-ish: points at user 0 (queue check handles empty case)
            else:
                action.append(env.controlled_users.index(u))
        return np.array(action)
    return policy_fn


def make_random_policy():
    def policy_fn(env, obs):
        return env.action_space.sample()
    return policy_fn


def make_ppo_policy(model):
    def policy_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy_fn


if __name__ == "__main__":
    model = PPO.load("results/trained_models/ppo_scheduler")
    model_ft = PPO.load("results/trained_models/ppo_finetuned")

    policies = {
        "Random": make_random_policy(),
        "Round Robin": make_baseline_policy(RoundRobinScheduler()),
        "Max-SINR": make_baseline_policy(MaxSINRScheduler()),
        "Proportional Fair": make_baseline_policy(ProportionalFairScheduler()),
        "PPO (from scratch)": make_ppo_policy(model),
        "PPO (BC warm-start + fine-tune)": make_ppo_policy(model_ft),
    }

    results = {}
    for name, policy_fn in policies.items():
        results[name] = eval_policy_on_env(policy_fn)
        tp, fair, lat = results[name]["throughput_mbps"], results[name]["fairness"], results[name]["latency_slots"]
        print(f"{name:20s} | throughput: {tp[0]:6.2f} +/- {tp[1]:4.2f} Mbps | "
              f"fairness: {fair[0]:.3f} +/- {fair[1]:.3f} | latency: {lat[0]:6.2f} +/- {lat[1]:5.2f} slots")

    # bar chart comparison
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ["throughput_mbps", "fairness", "latency_slots"],
        ["Total throughput (Mbps, higher better)", "Jain's fairness (higher better)", "Avg latency (slots, lower better)"],
    ):
        means = [results[n][key][0] for n in names]
        stds = [results[n][key][1] for n in names]
        colors = ["#888888", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
        ax.bar(names, means, yerr=stds, color=colors[:len(names)], capsize=4)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig("results/figures/phase6_agent_vs_baselines.png", dpi=120)
    print("Saved comparison plot.")

    # training curve comparison: from-scratch vs BC warm-start + fine-tune
    data = np.load("results/trained_models/training_curve.npz")
    data_ft = np.load("results/trained_models/finetune_curve.npz")
    plt.figure(figsize=(7, 4.5))
    plt.plot(data["timesteps"], data["rewards"], label="PPO from scratch")
    plt.plot(data_ft["timesteps"], data_ft["rewards"], label="PPO (BC warm-start + fine-tune)")
    plt.xlabel("training timesteps")
    plt.ylabel("mean episode reward")
    plt.title("Training curves: from-scratch vs. behavior-cloning warm-start")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/training_curve.png", dpi=120)
    print("Saved training curve comparison plot.")
