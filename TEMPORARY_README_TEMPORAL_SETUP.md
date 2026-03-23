# Temporary README: Temporal Graph Setup

This documents the current experimental workflow for the temporal GNN + transformer setup.
It is intentionally separate from the original pipeline scripts.

## What exists right now

- Original pipeline (unchanged): `run_pipeline.py`, `data/graph_dataset.py`
- Rich feature/data generation (separate): `run_rich_pipeline.py`, `data/graph_dataset_rich.py`
- Context-only + CSV supervision pipeline (separate):
  - `build_temporal_training_data.py`
  - `train_temporal_from_csv.py`

## Current training design

- Chronological time bins: one graph per day (`time_idx`)
- Train/test split: first 80% time steps train, last 20% test
- Graph stores stats only (no edge label supervision used during training)
- Supervision comes from CSV rows (`temporal_training_samples.csv`)
- Per time step `t`:
  - Run hetero `GATv2Conv` on graph at `t`
  - Build temporal context via transformer from up to 5 previous hidden states
  - Train on all PA rows at `t` (either all-at-once or mini-batched)
- Loss: cross entropy (optionally class-weighted)

## Data artifacts

Under `data/logs/`:

- `statcast_pitches.csv`: raw pitch data cache
- `graph_dataset_context_only.pkl`: graphs + metadata, stats-only
- `temporal_training_samples.csv`: one row per PA with labels/indices
- `test_predictions_temporal.csv`: final test predictions from trained model

## Build data

```bash
cd /Users/paul/Desktop/MLB_predictions
python build_temporal_training_data.py
```

Outputs:
- `data/logs/graph_dataset_context_only.pkl`
- `data/logs/temporal_training_samples.csv`

## Train model

Basic:

```bash
python train_temporal_from_csv.py --device cpu
```

CPU-friendly mini-batch per time step:

```bash
python train_temporal_from_csv.py \
  --device cpu \
  --epochs 20 \
  --pa-batch-size 128
```

Lighter class imbalance penalty example:

```bash
python train_temporal_from_csv.py \
  --device cpu \
  --class-weighting manual \
  --manual-class-weights "1.0,1.1,1.1,0.95"
```

Notes:
- `--class-weighting` options: `none`, `inverse_freq`, `manual`
- `--manual-class-weights` must match `label_to_idx` index order

## Predictions CSV output

After training, script writes:

- `data/logs/test_predictions_temporal.csv` (default)

Override path:

```bash
python train_temporal_from_csv.py --predictions-csv data/logs/my_preds.csv
```

Columns include:
- sample metadata (`time_idx`, game/date IDs, pitcher/batter IDs)
- true target (`target_idx`, `target_class`)
- prediction (`pred_idx`, `pred_class`, `correct`)
- per-class probabilities (`prob_<class_name>`)

## Known assumptions/limitations

- Time step is daily, not intra-game sequence.
- Temporal context window defaults to 5 previous steps.
- ERA/FIP features are approximated from available PA-level columns.
- Current trainer does not yet save model checkpoints.

## Next practical additions (optional)

- Add model checkpoint save/load
- Add validation split from train timeline for early stopping
- Add dedicated `predict_pa_probs.py` for single (pitcher, batter, t) inference

