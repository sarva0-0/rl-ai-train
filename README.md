# rl-ai-train

Train a program to land a rocket in a simulator. Look at each saved version in a web page.

## Purpose

This project shows how a reward teaches a program to land a rocket.

The project has two parts:

- **Training.** The program tries to land the rocket many times. It gets points for good behavior and loses points for bad behavior.
- **Viewer.** A web page shows each saved version of the program. You can see what it did and why it failed.

> **NOTE:** This project is a demonstration. It is not a landing system for a real rocket.

## Terms

Each term has one meaning in this document.

| Term | Meaning |
|---|---|
| Policy | The neural network that chooses a command from the rocket state. |
| Command | One of four choices: none, main engine, rotate nose left, rotate nose right. The code calls this an action. |
| Reward | The points the program gets after each step. |
| Episode | One landing attempt, from the start to the end. |
| Iteration | One update of the policy. One iteration uses 16,384 steps. |
| Checkpoint | A saved copy of the policy at one point in training. |
| Seed | A number that sets the start conditions of an episode. The same seed gives the same start. |
| Replay | A recorded episode that you can play in the viewer. |

## Requirements

- A Mac computer. The steps in this document are for macOS.
- Homebrew. Get it from [brew.sh](https://brew.sh).
- Python 3.12.
- Free disk space for PyTorch. PyTorch is large.

## Install

1. Open Terminal.
2. Install Python and SWIG:

   ```bash
   brew install python@3.12 swig
   ```

3. Go to the project folder:

   ```bash
   cd rl-ai-train
   ```

4. Make a virtual environment:

   ```bash
   python3.12 -m venv .venv
   ```

5. Start the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

6. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

Start the virtual environment again each time you open a new Terminal window.

## Run

Run each command in the project folder. Keep the virtual environment active.

1. Run the test:

   ```bash
   pytest -q
   ```

   The test is correct when the output shows `1 passed`.

2. Train the policy:

   ```bash
   python train.py
   ```

   Training takes about 10 to 20 minutes on a recent Mac. This time is an estimate.

3. Record the replays:

   ```bash
   python record_replays.py
   ```

4. Open the viewer:

   ```bash
   open viewer.html
   ```

> **NOTE:** Keep `replays.js` in the same folder as `viewer.html`. The viewer does not work without it.

### Watch training (optional)

1. Open a second Terminal window.
2. Start the virtual environment in that window.
3. Run this command:

   ```bash
   tensorboard --logdir tb
   ```

4. Open `http://localhost:6006` in a web browser.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the policy. Saves checkpoints and the log file. |
| `record_replays.py` | Tests each checkpoint. Writes `replays.js`. |
| `viewer.html` | Shows the replays. |
| `test_shaping.py` | Tests the reward shaping. |
| `requirements.txt` | Lists the Python libraries. |

Training and recording make these files and folders:

| Item | Content |
|---|---|
| `checkpoints/` | One checkpoint for every 50,000 steps, and one before training. |
| `logs/training_log.jsonl` | One log line for each iteration. |
| `tb/` | Data for TensorBoard. |
| `lander_ppo.zip` | The final policy. |
| `replays.js` | The replay data for the viewer. |

## How the reward works

The reward has these parts. `train.py` contains all of them.

- **Progress.** The program gets points when the rocket gets closer to the pad, slows down, and tilts less. This part uses potential-based shaping. `test_shaping.py` checks it.
- **Time cost.** The program loses 0.1 points each step.
- **Engine cost.** The program loses 0.3 points for each main engine step. It loses 0.03 points for each side engine step.
- **Offset cost.** The program loses points each step when the rocket is away from the pad center. The loss is `0.02 × (e^(4 × |x|) − 1)`, where `x` is the offset. The loss grows quickly as the offset grows.
- **End bonus.** The program gets 100 points for a safe landing. It loses 100 points for a crash.

The policy never receives instructions on how to fly. It learns which commands give the most points.

> **CAUTION:** A policy can find ways to get points that you did not intend. Always look at the replays. Do not use the score alone.

## Outcomes

`record_replays.py` tests each checkpoint with 50 seeds. It gives each episode one of these outcomes.

| Outcome | Meaning |
|---|---|
| Landed on pad | The lander came to rest. The offset from the pad center is 0.2 or less. |
| Landed off pad | The lander came to rest. The offset is more than 0.2. |
| Crash | The lander body touched the ground. |
| Out of bounds | The lander left the screen. |
| Timeout | The lander did not come to rest in 1,000 steps. |

The simulator counts a landing at any position and any tilt. The viewer shows the final tilt and leg contact for each replay.

## Viewer

The viewer shows 5 replays for each checkpoint. Use these controls:

- **Checkpoint.** Select a saved version of the policy.
- **Replay.** Select a seed. The name of the outcome shows next to the seed.
- **Play and slider.** Play the replay, or move to a frame.
- **Chart.** Select a point on the chart to select a checkpoint.
- **Jump buttons.** Go to the first checkpoint with an on-pad landing. Go to the first checkpoint with 80% on-pad landings. Go to the checkpoint with the best mean score.
- **Filter.** Hide the checkpoints that have no landings.

The viewer shows a burst of particles after a crash. It shows stars after a landing. The viewer does not show them if your system asks for reduced motion.

The viewer draws the lander at the simulator scale. The shapes of the legs and the body are simplified.

## Log file

`logs/training_log.jsonl` has one JSON object on each line. Each line is one iteration.

| Field | Meaning |
|---|---|
| `iteration` | The iteration number. |
| `steps` | The total number of steps so far. |
| `episodes` | The total number of finished episodes. |
| `mean_score` | The mean simulator score of the last 100 episodes. |
| `landing_rate` | The fraction of the last 100 episodes that ended in a landing. |

## Settings

Change a setting, then run `train.py` again.

| Setting | File | Meaning |
|---|---|---|
| `TOTAL_STEPS` | `train.py` | The number of training steps. |
| `CHECKPOINT_EVERY` | `train.py` | The number of steps between checkpoints. |
| `CENTER_PENALTY` | `train.py` | The scale of the offset cost. |
| `CENTER_RATE` | `train.py` | The growth rate of the offset cost. |
| `SEEDS` | `record_replays.py` | The number of seeds tested for each checkpoint. |
| `FULL` | `record_replays.py` | The number of replays saved for each checkpoint. |

> **NOTE:** If the rocket flies off the screen too often, decrease `CENTER_PENALTY`.

## Known limits

- The simulator is two-dimensional and simple.
- The simulator has no wind and no sensor noise.
- One training run gives one result. Results change between runs.
- The policy can find ways to get points that you did not intend.
- The viewer shape of the lander is simplified.
- The project is not tested on Windows or Linux.
