import sys
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.network_env import NetworkSchedulingEnv

TOTAL_TIMESTEPS = 150_000


class RewardLogger(BaseCallback):
    """Logs mean episode reward periodically so we can plot a training curve."""
    def __init__(self, log_every=1000):
        super().__init__()
        self.log_every = log_every
        self.rewards = []
        self.timesteps = []

    def _on_step(self):
        if self.n_calls % self.log_every == 0:
            ep_info_buf = self.model.ep_info_buffer
            if len(ep_info_buf) > 0:
                mean_r = np.mean([ep["r"] for ep in ep_info_buf])
                self.rewards.append(mean_r)
                self.timesteps.append(self.num_timesteps)
                print(f"timestep {self.num_timesteps}: mean episode reward = {mean_r:.3f}")
        return True


def main():
    env = Monitor(NetworkSchedulingEnv(n_users_per_cell=8, n_rb=3, episode_len=200, seed=0))
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=256,
                learning_rate=1e-4, gamma=0.97, ent_coef=0.01, clip_range=0.2,
                n_epochs=10, seed=0)

    logger = RewardLogger(log_every=2048)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger)
    model.save("results/trained_models/ppo_scheduler")

    np.savez("results/trained_models/training_curve.npz",
             timesteps=np.array(logger.timesteps), rewards=np.array(logger.rewards))
    print("Training complete. Model + training curve saved.")


if __name__ == "__main__":
    main()
