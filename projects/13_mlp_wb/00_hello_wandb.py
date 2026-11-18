"""
Minimal W&B smoke test 


-- not part of the real pipeline. Just enough
to see `wandb.init`, `wandb.log`, and `run.finish` produce one run in the
dashboard before building anything real on top of it.
"""

import random
import time

import wandb

run = wandb.init(
    entity="tekim",
    project="learn-bioml",
    name="hello-world", # the run's title in the dashboard 
    config={ #  hyperparameters and metadat; shown in the run's "Overview" tab
        "learning_rate": 0.01, 
        "note": "smoke test"
    },
)

for step in range(20):
    fake_loss = 1.0 / (step + 1) + random.uniform(0, 0.05)
    wandb.log({"loss": fake_loss, "step": step})  # one point per call, plotted live
    time.sleep(0.1)

run.finish()
print(f"\nDone. View this run at: {run.url}")
