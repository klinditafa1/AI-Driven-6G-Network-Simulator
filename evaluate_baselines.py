import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from core.scenario import make_hex_grid_base_stations, place_random_users, associate_users
from core.simulator import NetworkSim
from baselines.round_robin import RoundRobinScheduler
from baselines.max_sinr import MaxSINRScheduler
from baselines.proportional_fair import ProportionalFairScheduler

N_STEPS = 300
RB_BANDWIDTH_HZ = 1e6
DT_S = 1e-3


def build_scenario(seed=42, n_users=40, n_rb=3):
    rng = np.random.default_rng(seed)
    bases = make_hex_grid_base_stations(n_rings=1, cell_radius_m=250, n_rb=n_rb)
    users = place_random_users(n_users, area_radius_m=500, demand_range_mbps=(0.5, 5.0), rng=rng)
    associate_users(bases, users)
    return bases, users


def run_scheduler(name, scheduler_factory, seed=42):
    bases, users = build_scenario(seed=seed)
    sim = NetworkSim(bases, users, rb_bandwidth_hz=RB_BANDWIDTH_HZ, dt_s=DT_S, max_queue_bits=10e6)
    sched = scheduler_factory()
    schedulers = {bs.id: sched for bs in bases}
    for _ in range(N_STEPS):
        sim.step(schedulers)
    hist = sim.history
    total_tp = np.mean([h["total_throughput_mbps"] for h in hist])
    fairness = np.mean([h["fairness"] for h in hist])
    latency = np.mean([h["avg_latency_slots"] for h in hist])
    dropped = np.sum([h["dropped_mbits"] for h in hist])
    print(f"{name:20s} | avg throughput: {total_tp:7.2f} Mbps | fairness: {fairness:.3f} "
          f"| avg latency: {latency:5.2f} slots | total dropped: {dropped:6.2f} Mbit")
    return hist


if __name__ == "__main__":
    results = {}
    results["Round Robin"] = run_scheduler("Round Robin", RoundRobinScheduler)
    results["Max-SINR"] = run_scheduler("Max-SINR", MaxSINRScheduler)
    results["Proportional Fair"] = run_scheduler("Proportional Fair", ProportionalFairScheduler)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("total_throughput_mbps", "Total throughput (Mbps)"),
               ("fairness", "Jain's fairness index"),
               ("avg_latency_slots", "Avg queue latency (slots)")]
    for ax, (key, title) in zip(axes, metrics):
        for name, hist in results.items():
            vals = [h[key] for h in hist]
            ax.plot(vals, label=name, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/phase3_baseline_comparison.png", dpi=120)
    print("Saved comparison plot.")
