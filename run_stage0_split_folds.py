import pandas as pd
import numpy as np
import os
import argparse
import sys
import traceback

from config import SEED, HOLDOUT_ENTITY_FRACTION, CHRONO_SPLIT_RATIO


def split_dataset_forecasting(dataset_name, data_path, horizon,
                              holdout_entity_fraction, chrono_split_ratio,
                              seed, split_dir):
    """
    MARIO forecasting split (replaces the FCPM K-fold / GroupKFold split).

    Hybrid chronological + entity-holdout split. There is no classification,
    so there are no folds and no stratification.

    Two-axis partition:
      1. Entity axis (new-entity generalization): a seeded random
         `holdout_entity_fraction` of entities are held out entirely and never
         appear in training. They form the `new_entity` test regime.
      2. Time axis (temporal forecasting): the remaining "train entities" are
         cut chronologically at `chrono_split_ratio` of their time span. Their
         early portion is training; their late portion is the `seen_future`
         test regime.

    Embargo (correctness-critical): for a train entity, the last `horizon`
    time units before the cut are dropped from training. A training example at
    time t has target t + horizon, so without this gap the targets near the cut
    would leak into the test region. Holdout entities need no embargo.

    Seen-entity context: to score a seen entity's future slice at time t, we
    still need its pre-cut history to know which TIRP prefixes are active at t.
    Therefore test.csv carries each seen entity's FULL timeline, with `cut_time`
    in the manifest marking where scoring begins. Holdout entities are scored
    across their whole timeline (after a downstream warm-up); their cut_time is
    left empty (NaN).

    Outputs (into split_dir):
      train.csv           # train-entities only, pre-cut rows minus the embargo
      test.csv            # every entity's full timeline (context for seen,
                          #   full series for holdout)
      split_manifest.csv  # EntityID, role, test_regime, cut_time
    """
    print(f"--- Starting Stage 0: Forecasting Split for Dataset: {dataset_name} ---")
    print(f"Data Path: {data_path}")
    print(f"Horizon (embargo): {horizon}")
    print(f"Holdout entity fraction: {holdout_entity_fraction}")
    print(f"Chronological split ratio: {chrono_split_ratio}")
    print(f"Seed: {seed}")
    print(f"Output split directory: {split_dir}")

    try:
        # --- Input validation ---
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            sys.exit(1)
        if not (0.0 <= holdout_entity_fraction < 1.0):
            print(f"ERROR: holdout_entity_fraction must be in [0.0, 1.0), got {holdout_entity_fraction}")
            sys.exit(1)
        if not (0.0 < chrono_split_ratio < 1.0):
            print(f"ERROR: chrono_split_ratio must be in (0.0, 1.0), got {chrono_split_ratio}")
            sys.exit(1)
        if horizon < 0:
            print(f"ERROR: horizon must be >= 0, got {horizon}")
            sys.exit(1)

        # --- Load data ---
        print("Loading data...")
        data = pd.read_csv(data_path)
        print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

        if 'EntityID' not in data.columns:
            print("ERROR: Required column 'EntityID' not found in data.")
            sys.exit(1)

        # The pipeline uses Hugobot long format (EntityID, TemporalPropertyID,
        # TimeStamp, TemporalPropertyValue). Support 'TimeStamp' (long) and fall
        # back to 'Time' (wide) for genericity.
        time_col = 'TimeStamp' if 'TimeStamp' in data.columns else (
            'Time' if 'Time' in data.columns else None)
        if time_col is None:
            print("ERROR: No time column found (expected 'TimeStamp' or 'Time').")
            sys.exit(1)

        # Drop class rows (TemporalPropertyID == -1). These are per-entity
        # classification labels from the original FCPM dataset and have no role
        # in forecasting. Removing them here keeps the rest of the pipeline
        # class-free: Hugobot never sees a class map, so no per-class KL files
        # are produced downstream.
        if 'TemporalPropertyID' in data.columns:
            n_before = len(data)
            data = data[data['TemporalPropertyID'] != -1]
            n_dropped = n_before - len(data)
            if n_dropped:
                print(f"Dropped {n_dropped} class-label rows "
                      f"(TemporalPropertyID == -1); forecasting has no classes.")

        # --- Entity axis: seeded holdout partition ---
        # Sort entity ids first so the permutation is independent of row order.
        entities = np.array(sorted(data['EntityID'].unique()))
        n_entities = len(entities)
        rng = np.random.default_rng(seed)
        entities = entities[rng.permutation(n_entities)]

        n_holdout = int(round(holdout_entity_fraction * n_entities))
        # Guarantee at least one train entity remains.
        n_holdout = min(n_holdout, n_entities - 1)
        holdout_entities = set(entities[:n_holdout])
        train_entities = set(entities[n_holdout:])

        print(f"Entities: {n_entities} total -> {len(train_entities)} train, "
              f"{len(holdout_entities)} holdout")

        # --- Time axis: per-train-entity chronological cut ---
        # cut_time = t_min + ratio * (t_max - t_min).
        train_measure = data[data['EntityID'].isin(train_entities)]
        t_min = train_measure.groupby('EntityID')[time_col].min()
        t_max = train_measure.groupby('EntityID')[time_col].max()
        cut_time = t_min + chrono_split_ratio * (t_max - t_min)
        cut_map = cut_time.to_dict()

        # Train rows: train entities, keep only rows whose target (t + horizon)
        # stays within the pre-cut region -> TimeStamp <= cut_time - horizon.
        train_mask = data['EntityID'].isin(train_entities) & (
            data[time_col] <= data['EntityID'].map(cut_map) - horizon
        )
        train_data = data[train_mask].sort_values(['EntityID', time_col])

        # Test rows: every entity's full timeline. Seen entities need their
        # pre-cut history for prefix context; holdout entities are fully unseen.
        test_data = data.sort_values(['EntityID', time_col])

        # Warn about train entities that ended up with no usable train rows
        # (e.g. very short timelines relative to the embargo).
        train_entities_with_rows = set(train_data['EntityID'].unique())
        empty_train_entities = train_entities - train_entities_with_rows
        if empty_train_entities:
            print(f"WARNING: {len(empty_train_entities)} train entities have no "
                  f"training rows after the embargo (timeline too short). They "
                  f"still contribute a seen_future test slice.")
        if train_data.empty:
            print("ERROR: No training rows produced. Check horizon / "
                  "chrono_split_ratio against the entities' timeline lengths.")
            sys.exit(1)

        # --- Manifest ---
        manifest_rows = []
        for e in train_entities:
            manifest_rows.append({
                'EntityID': e,
                'role': 'train',
                'test_regime': 'seen_future',
                'cut_time': cut_map.get(e, np.nan),
            })
        for e in holdout_entities:
            manifest_rows.append({
                'EntityID': e,
                'role': 'holdout',
                'test_regime': 'new_entity',
                'cut_time': np.nan,  # scored from start (after downstream warm-up)
            })
        manifest = pd.DataFrame(
            manifest_rows, columns=['EntityID', 'role', 'test_regime', 'cut_time']
        ).sort_values('EntityID')

        # --- Save ---
        os.makedirs(split_dir, exist_ok=True)
        train_file = os.path.join(split_dir, 'train.csv')
        test_file = os.path.join(split_dir, 'test.csv')
        manifest_file = os.path.join(split_dir, 'split_manifest.csv')

        print(f"  Train data shape: {train_data.shape} -> {os.path.basename(train_file)}")
        train_data.to_csv(train_file, index=False)
        print(f"  Test data shape: {test_data.shape} -> {os.path.basename(test_file)}")
        test_data.to_csv(test_file, index=False)
        print(f"  Manifest rows: {manifest.shape[0]} -> {os.path.basename(manifest_file)}")
        manifest.to_csv(manifest_file, index=False)

        print(f"--- Finished Stage 0: Forecasting Split for Dataset: {dataset_name} ---")

    except FileNotFoundError as e:
        print(f"ERROR: File not found during Stage 0 execution: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing expected column in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Stage 0 execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 0 (MARIO): hybrid chronological + entity-holdout "
                    "forecasting split."
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--horizon", required=True, type=float,
                        help="Forecast horizon; also used as the pre-cut embargo size.")
    parser.add_argument("--holdout_entity_fraction", type=float,
                        default=HOLDOUT_ENTITY_FRACTION,
                        help="Fraction of entities held out entirely (new-entity regime).")
    parser.add_argument("--chrono_split_ratio", type=float,
                        default=CHRONO_SPLIT_RATIO,
                        help="Per-entity past/future cut, as a fraction of time span.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for the entity holdout partition.")
    parser.add_argument("--split_dir", required=True,
                        help="Output directory for the split (reuses the former fold_1/ slot).")
    args = parser.parse_args()

    split_dataset_forecasting(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        horizon=args.horizon,
        holdout_entity_fraction=args.holdout_entity_fraction,
        chrono_split_ratio=args.chrono_split_ratio,
        seed=args.seed,
        split_dir=args.split_dir,
    )

    sys.exit(0)
