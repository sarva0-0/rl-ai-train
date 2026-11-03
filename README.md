# rl-ai-train

Reward-shaped PPO training for the Gymnasium `LunarLander-v3` environment, with a replay viewer that audits each saved checkpoint.

## Overview

The project trains a policy with a custom reward and then examines what the policy does. It has three parts:

- **Training.** `train.py` replaces the simulator reward with a shaped reward and trains a PPO policy.
- **Evaluation.** `record_replays.py` tests every checkpoint on fixed seeds and classifies each episode.
- **Audit.** `viewer.html` replays stored episodes frame by frame. It shows outcome, tilt, speed and leg contact.

The simulator score alone hides failures such as off-pad landings and engine use at rest. The viewer exists to expose them.

> [!NOTE]
> This is a research demonstrator. It is not a landing system for a real vehicle.

## Method

**Policy.** Stable-Baselines3 PPO with the default `MlpPolicy`. The policy reads the 8-value simulator state and outputs one of 4 commands: none, main engine, rotate nose left, rotate nose right.

**Training setup.**

| Parameter | Value |
|---|---|
| Parallel environments | 16 |
| Total steps | 1,000,000 |
| Steps per iteration | 16,384 (16 environments × 1,024) |
| Batch size / epochs | 64 / 4 |
| Discount (γ) / GAE λ | 0.999 / 0.98 |
| Entropy coefficient | 0.01 |
| Checkpoint interval | 50,000 steps |

**Reward.** The wrapper `ShapedLander` in `train.py` computes the reward from these terms:

| Term | Definition |
|---|---|
| Progress | `γ·Φ(s′) − Φ(s)`, with `Φ = −100·distance − 100·speed − 100·|angle| + 10·leg contacts` |
| Time | −0.1 per step |
| Engines | −0.3 per main-engine step, −0.03 per side-engine step |
| Offset | `−0.02·(e^(4·|x|) − 1)` per step, where `x` is the offset from the pad center |
| Terminal | +100 for a landing, −100 for a crash or leaving the screen |

The progress term is potential-based. `test_shaping.py` checks that its discounted sum telescopes as theory requires. The time, engine and offset terms change the optimal policy by design.

## Quick start

Requirements: Python 3.12 and SWIG (SWIG builds the Box2D physics engine). The commands below use macOS and Homebrew. The project is not tested on Windows or Linux.

```bash
brew install python@3.12 swig
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                  # reward shaping test
python train.py            # train, save checkpoints and log
python record_replays.py   # evaluate checkpoints, write replays.js
python make_figures.py     # write figures/
open viewer.html           # keep replays.js in the same folder
```

Monitor training with `tensorboard --logdir tb`.

## Repository layout

| Path | Purpose |
|---|---|
| `train.py` | Reward wrapper, PPO training, checkpoints, training log |
| `record_replays.py` | Checkpoint evaluation and replay export |
| `viewer.html` | Replay viewer. Reads `replays.js`. |
| `make_figures.py` | Training-score and checkpoint-outcome charts |
| `test_shaping.py` | Test for the potential-based shaping term |

Generated at run time: `checkpoints/`, `logs/training_log.jsonl`, `tb/`, `lander_ppo.zip`, `replays.js`, `figures/`.

## Evaluation protocol

`record_replays.py` flies each checkpoint on seeds 0 to 49 with a deterministic policy. It stores the full trajectory for seeds 0 to 4. Every episode receives one outcome:

| Outcome | Definition |
|---|---|
| Landed on pad | The lander came to rest with `|x| ≤ 0.2`. |
| Landed off pad | The lander came to rest with `|x| > 0.2`. |
| Crash | The lander body touched the ground. |
| Out of bounds | The lander left the screen. |
| Timeout | The lander did not come to rest within 1,000 steps. |

The simulator counts a landing at any position and any tilt. The viewer therefore reports final tilt, speed and leg contact for each replay.

## Replay viewer

- Select a checkpoint and a stored replay. Play the replay or move to any frame.
- Read telemetry for each frame: position, velocity, tilt, leg contact and command.
- Read the outcome panel for the final state and the reason for the outcome.
- Use the landing-rate chart or the jump buttons to find the checkpoint where landings first appear.
- Filter the checkpoint table to show only checkpoints with landings.

The drawing uses the simulator scale. Hull and leg shapes are simplified.

## Training log

`logs/training_log.jsonl` has one JSON object for each PPO iteration:

| Field | Meaning |
|---|---|
| `iteration` | Iteration number |
| `steps` | Total steps so far |
| `episodes` | Total finished episodes |
| `mean_score` | Mean simulator score of the last 100 episodes |
| `landing_rate` | Fraction of the last 100 episodes that ended in a landing |

Training scores come from the stochastic policy. Checkpoint evaluation uses the deterministic policy on fixed seeds. Do not compare the two directly.

## Configuration

| Setting | File | Effect |
|---|---|---|
| `TOTAL_STEPS` | `train.py` | Training length |
| `CHECKPOINT_EVERY` | `train.py` | Steps between checkpoints |
| `CENTER_PENALTY`, `CENTER_RATE` | `train.py` | Scale and growth of the offset penalty |
| `SEEDS`, `FULL` | `record_replays.py` | Test seeds per checkpoint, and stored replays per checkpoint |

A large offset penalty can make an early crash cheaper than drifting. Reduce `CENTER_PENALTY` if the lander leaves the screen more often.

## Limitations

- The simulator is 2D and simple. It has no wind and no sensor noise.
- `train.py` sets no random seed. Results change between runs.
- A policy can score points in ways the reward designer did not intend. Inspect replays, not only scores.
- The evaluation uses 50 seeds for each checkpoint. Small differences between checkpoints are not significant.
