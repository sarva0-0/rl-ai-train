"""Train a PPO policy to land the LunarLander rocket with a shaped reward."""
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

N_ENVS = 16
TOTAL_STEPS = 1_000_000
CHECKPOINT_EVERY = 50_000  # Steps between saved checkpoints.
CENTER_PENALTY = 0.02      # Scale of the per-frame offset penalty.
CENTER_RATE = 4.0          # Growth rate of the penalty with offset.


class ShapedLander(gym.Wrapper):
    """Replace the simulator reward with a shaped reward."""

    gamma = 0.999  # Keep equal to the PPO gamma.

    @staticmethod
    def _potential(obs):
        x, y, vx, vy, angle, _, leg1, leg2 = obs
        return (-100 * np.hypot(x, y) - 100 * np.hypot(vx, vy)
                - 100 * abs(angle) + 10 * (leg1 + leg2))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev, self.official = self._potential(obs), 0.0
        return obs, info

    def step(self, action):
        obs, base, term, trunc, info = self.env.step(action)
        self.official += base  # Keep the simulator score for the log.
        pot = self._potential(obs)
        shaping = self.gamma * pot - self.prev
        reward = shaping
        self.prev = pot
        reward -= 0.1  # Time cost.
        if action == 2:
            reward -= 0.3  # Main engine cost.
        elif action in (1, 3):
            reward -= 0.03  # Side engine cost.
        reward -= CENTER_PENALTY * (np.exp(CENTER_RATE * abs(obs[0])) - 1)  # Grows exponentially with offset from the pad.
        if term:
            if base >= 100:
                reward += 100  # Safe landing.
            elif base <= -100:
                reward -= 100  # Crash.
        info["shaping"] = shaping
        if term or trunc:
            info["official"] = self.official
            info["landed"] = bool(term and base >= 100)
        return obs, reward, term, trunc, info


class Progress(BaseCallback):
    """Write one log line for each PPO update (one iteration)."""

    def __init__(self, path="logs/training_log.jsonl"):
        super().__init__()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        self.path, self.rows, self.iteration = path, [], 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "official" in info:
                self.rows.append((info["official"], info["landed"]))
        return True

    def _on_rollout_end(self):
        self.iteration += 1
        if not self.rows:
            return
        recent = np.array(self.rows[-100:], dtype=float)
        record = {
            "iteration": self.iteration,
            "steps": self.num_timesteps,
            "episodes": len(self.rows),
            "mean_score": round(float(recent[:, 0].mean()), 2),
            "landing_rate": round(float(recent[:, 1].mean()), 3),
        }
        self.logger.record("custom/official_score", record["mean_score"])
        self.logger.record("custom/landing_rate", record["landing_rate"])
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    env = make_vec_env("LunarLander-v3", n_envs=N_ENVS, wrapper_class=ShapedLander)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4,
                gamma=ShapedLander.gamma, gae_lambda=0.98, ent_coef=0.01,
                tensorboard_log="tb", verbose=1)
    model.save("checkpoints/lander_0_steps")  # Untrained version.
    checkpoints = CheckpointCallback(save_freq=CHECKPOINT_EVERY // N_ENVS,
                                     save_path="checkpoints", name_prefix="lander")
    model.learn(TOTAL_STEPS, callback=CallbackList([Progress(), checkpoints]))
    model.save("lander_ppo")
