"""Make figures 9.1 and 9.2 from the log file and the replay data."""
import json
import os

import matplotlib
matplotlib.use("Agg")  # Save files. Do not open a window.
import matplotlib.pyplot as plt

LOG = "logs/training_log.jsonl"
REPLAYS = "replays.js"
OUT = "figures"

for path in (LOG, REPLAYS):
    if not os.path.exists(path):
        raise SystemExit(f"File {path} not found. Run train.py and record_replays.py first.")
os.makedirs(OUT, exist_ok=True)

# Figure 9.1: mean official score for each PPO update.
with open(LOG) as fh:
    rows = [json.loads(line) for line in fh if line.strip()]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([r["iteration"] for r in rows], [r["mean_score"] for r in rows], color="#1f4e99", lw=1.5)
ax.set_xlabel("PPO update (iteration)")
ax.set_ylabel("Mean official score (last 100 episodes)")
ax.set_title("Figure 9.1  Training mean official score")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_9_1_training_score.png", dpi=200)
plt.close(fig)

# Figure 9.2: outcomes for each checkpoint (stacked bars).
with open(REPLAYS) as fh:
    text = fh.read()
checkpoints = json.loads(text[text.index("{"): text.rindex("}") + 1])["checkpoints"]
labels = [f"{c['steps'] // 1000}k" for c in checkpoints]
get = lambda c, k: c["counts"].get(k, 0)
series = [
    ("Landed on pad", [get(c, "LANDED") for c in checkpoints], "#2e8b57"),
    ("Landed off pad", [get(c, "LANDED_OFF_PAD") for c in checkpoints], "#d99a1f"),
    ("Crash", [get(c, "CRASH") for c in checkpoints], "#c0392b"),
    ("Out of bounds or timeout",
     [get(c, "OUT_OF_BOUNDS") + get(c, "TIMEOUT") for c in checkpoints], "#7f8c8d"),
]
fig, ax = plt.subplots(figsize=(10, 4.5))
bottom = [0] * len(checkpoints)
for name, values, color in series:
    ax.bar(labels, values, bottom=bottom, label=name, color=color)
    bottom = [b + v for b, v in zip(bottom, values)]
ax.set_xlabel("Checkpoint (training steps)")
ax.set_ylabel("Episodes (fixed test seeds)")
ax.set_title("Figure 9.2  Checkpoint evaluation results")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
plt.setp(ax.get_xticklabels(), rotation=45)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_9_2_checkpoint_outcomes.png", dpi=200)
plt.close(fig)
print(f"Saved two figures in {OUT}/")
