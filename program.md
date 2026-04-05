# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify the evaluation harness. The `evaluate_segmentation` function in `prepare.py` returns a dict. "f1" key in this dict is the ground truth metric.

**The goal is simple: get the Highest F1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful F1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 F1 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 F1 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Task explained

The task is **binary semantic segmentation** on Landslide4Sense patches:
- Pixel label `1`: landslide
- Pixel label `0`: non-landslide
- Input shape: `128x128`
- Input channels: `14`

Data split used in this assignment:
- Train: `../data/landSlide4Sense/TrainData/{img,mask}`
- Validation: `../data/landSlide4Sense/ValidData/{img,mask}`
- Test: `../data/landSlide4Sense/TestData/{img,test}`

## Primary metrics

This workflow is optimizing `F1` score.

Primary metrics:
- F1 Score
- IoU
- Precision
- Recall

Efficiency score:

$$
Z = 0.6 \cdot F1\% + 0.4 \cdot IoU\%
$$

## Rules and constraints

1. No data leakage.
2. Do not use test labels for training updates.
3. Threshold tuning is allowed only on validation.
4. Any model change must keep 14-channel input support.
5. Prefer stable improvements in F1/IoU (and Z), not just lower training loss.

## In-scope files

- `prepare.py`: local dataset paths, normalization stats, dataloaders, F1/IoU evaluation helpers.
- `train.py`: model architecture, losses, training loop, threshold search, checkpointing.
- `program.md`: this assignment-aligned execution policy.

## Run protocol

1. Validate dataset and stats:

```bash
uv run prepare.py
```

2. Train and evaluate:

```bash
uv run train.py
```

3. Capture final metrics from output:
- Validation: Precision, Recall, F1, IoU, Z
- Test: Precision, Recall, F1, IoU, Z
- Best threshold and trainable parameters

## Logging format (updated)

Use `results.tsv` with this header:

```tsv
commit	val_f1	val_iou	val_z	test_f1	test_iou	test_z	status	description
```

Suggested status values:
- `keep`: better F1/IoU or better Z with stable behavior
- `discard`: no meaningful improvement
- `crash`: failed run (OOM/bug)



## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^F1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If F1 improved (lower), you "advance" the branch, keeping the git commit
9. If F1 is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!