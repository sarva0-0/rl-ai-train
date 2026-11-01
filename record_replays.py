"""Record replay data for viewer.html. Run this after train.py."""
import glob
import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

SEEDS = 50  # Seeds tested for each checkpoint.
FULL = 5    # Seeds saved frame by frame.


def row(obs, action):
    return [round(float(v), 3) for v in obs[:5]] + [int(obs[6]), int(obs[7]), action]


def fly(model, env, seed):
    obs, _ = env.reset(seed=seed)
    frames, score, done = [row(obs, 0)], 0.0, False
    while not done:
        action = int(model.predict(obs, deterministic=True)[0])
        obs, reward, term, trunc, _ = env.step(action)
        score += reward
        frames.append(row(obs, action))
        done = term or trunc
    if term and reward >= 100:
        outcome = "LANDED" if abs(obs[0]) <= 0.2 else "LANDED_OFF_PAD"
    elif term and abs(obs[0]) >= 1:
        outcome = "OUT_OF_BOUNDS"
    elif term:
        outcome = "CRASH"
    else:
        outcome = "TIMEOUT"
    return outcome, score, frames


files = sorted(glob.glob("checkpoints/lander_*_steps.zip"),
               key=lambda p: int(p.split("_")[-2]))
env = gym.make("LunarLander-v3")
data = []
for path in files:
    steps, model = int(path.split("_")[-2]), PPO.load(path)
    counts, scores, runs = {}, [], []
    for seed in range(SEEDS):
        outcome, score, frames = fly(model, env, seed)
        counts[outcome] = counts.get(outcome, 0) + 1
        scores.append(score)
        if seed < FULL:
            runs.append({"seed": seed, "outcome": outcome,
                         "score": round(score, 1), "frames": frames})
    data.append({"steps": steps, "counts": counts,
                 "mean_score": round(float(np.mean(scores)), 1), "runs": runs})
    print(f"{steps:>9} steps  {counts}")

with open("replays.js", "w") as fh:
    fh.write("const REPLAYS = " + json.dumps({"checkpoints": data}, separators=(",", ":")) + ";\n")
