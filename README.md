# rl-ai-train

Train a program to land a rocket in a simulator. Look at each saved version in a web page.

## Purpose

This project shows how a reward teaches a program to land a rocket.

The project has two parts:

- **Training.** The program tries to land the rocket many times. It gets points for good behavior and loses points for bad behavior.
- **Viewer.** A web page shows each saved version of the program. You can see what it did and why it failed.

> **WARNING:** This project is a demonstration. Do not use it to control a real rocket or any real machine.

## Terms

Each term has one meaning in this document.

| Term | Meaning |
|---|---|
| Policy | The neural network that chooses a command. |
| Command | One of four choices: none, main engine, rotate nose left, rotate nose right. The code calls this an action. |
| Reward | The points the program gets after each step. |
| Episode | One landing attempt, from the start to the end. |
| Iteration | One update of the policy. One iteration uses 16,384 steps. |
| Checkpoint | A saved copy of the policy at one point in training. |
| Seed | A number that sets the start conditions of an episode. The same seed gives the same start. |
| Replay | A recorded episode that you can play in the viewer. |

## Requirements

- Python 3.12.
- SWIG. The Box2D physics engine needs it.
- A Mac with Homebrew. The steps below are for macOS. The project is not tested on Windows or Linux.

## Install

1. Install Python and SWIG:

   ```bash
   brew install python@3.12 swig
   ```

2. Make a virtual environment in the project folder:

   ```bash
   python3.12 -m venv .venv
   ```

3. Start the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

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

4. Make the charts:

   ```bash
   python make_figures.py
   ```

5. Open the viewer:

   ```bash
   open viewer.html
   ```

> **NOTE:** Keep `replays.js` in the same folder as `viewer.html`. The viewer does not work without it.

> **CAUTION:** `train.py` overwrites `logs/training_log.jsonl`. `record_replays.py` overwrites `replays.js`. Copy these files before you run the scripts again.

To watch training, run `tensorboard --logdir tb` in a second window. Open `http://localhost:6006` in a web browser.

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains the policy. Saves checkpoints and the log file. |
| `record_replays.py` | Tests each checkpoint. Writes `replays.js`. |
| `viewer.html` | Shows the replays. |
| `make_figures.py` | Makes two charts: training score and checkpoint results. |
| `test_shaping.py` | Tests the reward. |
| `requirements.txt` | Lists the Python libraries. |

The scripts make these files and folders:

| Item | Content |
|---|---|
| `checkpoints/` | One checkpoint for every 50,000 steps, and one before training. |
| `logs/training_log.jsonl` | One log line for each iteration. |
| `tb/` | Data for TensorBoard. |
| `lander_ppo.zip` | The final policy. |
| `replays.js` | The replay data for the viewer. |
| `figures/` | The two charts as image files. |

## How the reward works

`train.py` contains the reward. It has these parts:

- **Progress.** The program gets points when the rocket gets closer to the pad, slows down, and tilts less.
- **Time cost.** The program loses 0.1 points each step.
- **Engine cost.** The program loses 0.3 points for each main engine step. It loses 0.03 points for each side engine step.
- **Offset cost.** The program loses points each step when the rocket is away from the pad center. The loss grows quickly as the offset grows.
- **End bonus.** The program gets 100 points for a safe landing. It loses 100 points for a crash.

The policy does not receive instructions on how to fly. It learns which commands give the most points.

> **CAUTION:** A policy can find ways to get points that you did not intend. Always look at the replays. Do not use the score alone.

## Outcomes

`record_replays.py` tests each checkpoint with 50 seeds. Each episode gets one outcome:

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
- **Replay.** Select a seed. The outcome shows next to the seed.
- **Play and slider.** Play the replay, or move to a frame.
- **Chart.** Select a point on the chart to select a checkpoint.
- **Jump buttons.** Go to the first on-pad landing, the first checkpoint with 80% on-pad landings, or the best mean score.
- **Filter.** Hide the checkpoints that have no landings.

The viewer shows a burst of particles after a crash and stars after a landing. It does not show them if your system asks for reduced motion.

## Log file

`logs/training_log.jsonl` has one JSON object on each line. Each line is one iteration.

| Field | Meaning |
|---|---|
| `iteration` | The iteration number. |
| `steps` | The total number of steps so far. |
| `episodes` | The total number of finished episodes. |
| `mean_score` | The mean simulator score of the last 100 episodes. |
| `landing_rate` | The fraction of the last 100 episodes that ended in a landing. |

Training uses random choices. Checkpoint tests do not. Do not compare the scores from the two directly.

## Settings

Change a setting, then run the script again.

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

- The simulator is two-dimensional and simple. It has no wind and no sensor noise.
- `train.py` does not set a random seed. Results change between runs.
- One training run gives one result.
- Each checkpoint uses 50 test seeds. Small differences between checkpoints are not significant.
- The policy can find ways to get points that you did not intend.
- The shapes of the lander in the viewer are simplified.
