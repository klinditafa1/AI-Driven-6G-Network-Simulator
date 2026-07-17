import sys, os
sys.path.insert(0, '.')
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.network_env import NetworkSchedulingEnv

INCREMENT = 20_000
BC_MODEL_PATH = "results/trained_models/ppo_bc_warmstart.zip"
MODEL_PATH = "results/trained_models/ppo_finetuned.zip"
CURVE_PATH = "results/trained_models/finetune_curve.npz"


class RewardLogger(BaseCallback):
    def __init__(self, log_every=2048):
        super().__init__()
        self.log_every = log_every
        self.rewards = []
        self.timesteps = []

    def _on_step(self):
        if self.n_calls % self.log_every == 0:
            buf = self.model.ep_info_buffer
            if len(buf) > 0:
                mean_r = np.mean([ep["r"] for ep in buf])
                self.rewards.append(mean_r)
                self.timesteps.append(self.num_timesteps)
        return True


def main():
    env = Monitor(NetworkSchedulingEnv(n_users_per_cell=8, n_rb=3, episode_len=200, seed=0))

    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=env)
        print(f"Resuming fine-tune from {model.num_timesteps} timesteps")
    else:
        model = PPO.load(BC_MODEL_PATH, env=env)
        model.num_timesteps = 0  # restart the counter for the fine-tuning curve
        print("Starting fine-tune from BC warm-start")

    logger = RewardLogger(log_every=2048)
    model.learn(total_timesteps=INCREMENT, callback=logger, reset_num_timesteps=False)
    model.save(MODEL_PATH.replace(".zip", ""))

    if os.path.exists(CURVE_PATH):
        old = np.load(CURVE_PATH)
        all_timesteps = np.concatenate([old["timesteps"], np.array(logger.timesteps)])
        all_rewards = np.concatenate([old["rewards"], np.array(logger.rewards)])
    else:
        all_timesteps = np.array(logger.timesteps)
        all_rewards = np.array(logger.rewards)
    np.savez(CURVE_PATH, timesteps=all_timesteps, rewards=all_rewards)

    print(f"Now at {model.num_timesteps} total fine-tune timesteps. "
          f"Last mean episode reward: {logger.rewards[-1] if logger.rewards else 'n/a'}")


if __name__ == "__main__":
    main()
