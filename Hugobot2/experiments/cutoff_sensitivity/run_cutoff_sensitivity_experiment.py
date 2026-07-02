"""
Cutoff Sensitivity Experiment across 5-Fold CV for TID3 Variants
================================================================

For each (dataset x method x bins x fold) combination this script:
  1. Loads the fold-specific train CSV.
  2. Runs Hugobot2 abstraction with the requested TID3 variant.
  3. Saves states.csv / KL.txt / etc. under runs/<dataset>/<combo>/fold_<k>/.
  4. Extracts per-variable cutoffs **and** their index in the original sorted
     candidate pool (post-hoc, deterministic).
  5. Collects everything into a single cutoffs_sensitivity.csv with columns:
       dataset, combo, TemporalPropertyID, cutoff_idx,
       fold_1 .. fold_5            (cutoff values),
       fold_1_pool_idx .. fold_5   (candidate-pool position),
       fold_1_pool_size .. fold_5  (pool length for normalisation).

Run from this directory:
    cd Hugobot2/experiments/cutoff_sensitivity
    python run_cutoff_sensitivity_experiment.py
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Hugobot2.ta_package import TemporalAbstraction
from Hugobot2.ta_package import utils as ta_utils
from Hugobot2.ta_package.utils import generate_candidate_cutpoints, remove_na
from Hugobot2.ta_package.constants import TEMPORAL_PROPERTY_ID

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "falls_small",
        "fold_dir": "/sise/robertmo-group/Eldar/TIDE_FCPM/falls_small",
    },
    {
        "name": "diabetes",
        "fold_dir": "/sise/robertmo-group/Eldar/TIDE_FCPM/diabetes",
    },
    {
        "name": "icu",
        "fold_dir": "/sise/robertmo-group/Eldar/TIDE_FCPM/icu",
    },
    {
        "name": "ahe_small",
        "fold_dir": "/sise/robertmo-group/Eldar/TIDE_FCPM/ahe_small",
    },
]

METHODS = ["tid3", "tid3_c0longer", "tid3_c1longer"]
BINS_LIST = [2, 3, 4, 5]
INTERPOLATION_GAP = 1
NB_CANDIDATES = 100
N_FOLDS = 5

RUNS_DIR = os.path.join(HERE, "runs")
OUTPUT_CSV = os.path.join(HERE, "cutoffs_sensitivity.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tid3_method_config(method_name: str, bins: int) -> dict:
    scoring_method, duration_pref, mv, nb_candidates, _, _ = ta_utils.parse_tid3_config(method_name)
    tid3_cfg = {
        "method": "tid3",
        "bins": bins,
        "scoring_method": scoring_method,
        "duration_preference": duration_pref,
        "nb_candidates": nb_candidates if nb_candidates is not None else NB_CANDIDATES,
        "multivariate_refinement": mv,
    }
    return {"default": [tid3_cfg]}


def combo_string(method_name: str, bins: int) -> str:
    return f"abs_b={bins}-{method_name}-ig={INTERPOLATION_GAP}"


def find_pool_index(pool: list, value: float) -> int:
    """Return the index of the closest element in *pool* to *value*."""
    if not pool:
        return -1
    arr = np.asarray(pool)
    idx = int(np.argmin(np.abs(arr - value)))
    return idx


# ---------------------------------------------------------------------------
# Run one (dataset, method, bins, fold) abstraction
# ---------------------------------------------------------------------------

def run_single_fold(dataset: dict, method_name: str, bins: int, fold: int):
    """
    Run abstraction for one fold.  Returns a list of dicts, one per
    (TemporalPropertyID, cutoff_idx), with cutoff value, pool index, and pool
    size.  Returns [] on error.
    """
    combo = combo_string(method_name, bins)
    out_dir = os.path.join(RUNS_DIR, dataset["name"], combo, f"fold_{fold}")
    train_path = os.path.join(dataset["fold_dir"], f"fold_{fold}", "train.csv")

    if not os.path.isfile(train_path):
        print(f"  [skip] train file not found: {train_path}")
        return []

    df = pd.read_csv(train_path, low_memory=False)

    # -- Run abstraction (skip if already done) --
    states_path = os.path.join(out_dir, "states.csv")
    if os.path.exists(states_path):
        print(f"  [skip] {dataset['name']}/{combo}/fold_{fold}: states.csv exists")
    else:
        os.makedirs(out_dir, exist_ok=True)
        method_config = build_tid3_method_config(method_name, bins)
        ta = TemporalAbstraction(df)
        ta.apply(
            method_config=method_config,
            per_entity=False,
            split_test=False,
            save_output=True,
            output_dir=out_dir,
            max_gap=INTERPOLATION_GAP,
        )

    # -- Extract cutoffs from states.csv --
    cutoffs_per_tpid = _extract_cutoffs_from_states(states_path)

    # -- Prepare class-free data for candidate-pool regeneration --
    # Replicate the same cleanup that core.py applies before calling TID3.fit()
    train_data = df[df[TEMPORAL_PROPERTY_ID] != -1].copy()
    train_data = remove_na(train_data)

    # -- For each variable, recover pool index of every cutoff --
    rows = []
    for tpid, cutoffs in cutoffs_per_tpid.items():
        group = train_data[train_data[TEMPORAL_PROPERTY_ID] == tpid]
        pool = generate_candidate_cutpoints(group, NB_CANDIDATES)
        pool_size = len(pool)

        for sorted_idx, cutoff_val in enumerate(cutoffs):
            pool_idx = find_pool_index(pool, cutoff_val)
            rows.append({
                "dataset": dataset["name"],
                "combo": combo,
                "TemporalPropertyID": tpid,
                "cutoff_idx": sorted_idx,
                "fold": fold,
                "cutoff_value": cutoff_val,
                "pool_idx": pool_idx,
                "pool_size": pool_size,
            })

    return rows


def _extract_cutoffs_from_states(states_csv: str) -> dict:
    """
    Parse states.csv and return {tpid: [sorted cutoff values]}.

    states.csv rows look like:
        StateID, TemporalPropertyID, BinId, BinLow, BinHigh
    Cutoffs are the finite BinHigh values (== BinLow of the next bin).
    """
    if not os.path.isfile(states_csv):
        return {}

    sdf = pd.read_csv(states_csv)
    cutoffs_per_tpid = {}

    for tpid, grp in sdf.groupby("TemporalPropertyID"):
        grp = grp.sort_values("BinId")
        highs = grp["BinHigh"].values
        cuts = [float(h) for h in highs if np.isfinite(h)]
        cuts = sorted(set(cuts))
        if cuts:
            cutoffs_per_tpid[tpid] = cuts

    return cutoffs_per_tpid


# ---------------------------------------------------------------------------
# Pivot long → wide
# ---------------------------------------------------------------------------

def pivot_to_wide(long_rows: list) -> pd.DataFrame:
    """
    Pivot the long-form rows collected from all folds into the wide CSV format
    expected by the analysis script.
    """
    if not long_rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(long_rows)

    idx_cols = ["dataset", "combo", "TemporalPropertyID", "cutoff_idx"]

    # Cutoff values
    val_pivot = long_df.pivot_table(
        index=idx_cols, columns="fold", values="cutoff_value", aggfunc="first"
    )
    val_pivot.columns = [f"fold_{int(c)}" for c in val_pivot.columns]

    # Pool indices
    pidx_pivot = long_df.pivot_table(
        index=idx_cols, columns="fold", values="pool_idx", aggfunc="first"
    )
    pidx_pivot.columns = [f"fold_{int(c)}_pool_idx" for c in pidx_pivot.columns]

    # Pool sizes
    psz_pivot = long_df.pivot_table(
        index=idx_cols, columns="fold", values="pool_size", aggfunc="first"
    )
    psz_pivot.columns = [f"fold_{int(c)}_pool_size" for c in psz_pivot.columns]

    wide = val_pivot.join(pidx_pivot).join(psz_pivot).reset_index()
    wide = wide.sort_values(idx_cols).reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_BY_NAME = {d["name"]: d for d in DATASETS}


def main():
    parser = argparse.ArgumentParser(description="Cutoff sensitivity experiment")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASET_BY_NAME.keys()),
                        help="Run only this dataset (default: all)")
    args = parser.parse_args()

    if args.dataset:
        datasets = [DATASET_BY_NAME[args.dataset]]
        output_csv = os.path.join(HERE, f"cutoffs_sensitivity_{args.dataset}.csv")
    else:
        datasets = DATASETS
        output_csv = OUTPUT_CSV

    print(f"Experiment root : {HERE}")
    print(f"Runs dir        : {RUNS_DIR}")
    print(f"Output CSV      : {output_csv}")
    print(f"Datasets        : {[d['name'] for d in datasets]}")
    print(f"Methods         : {METHODS}")
    print(f"Bins            : {BINS_LIST}")
    print(f"Folds           : {N_FOLDS}")
    print(f"nb_candidates   : {NB_CANDIDATES}")

    total_runs = len(datasets) * len(METHODS) * len(BINS_LIST) * N_FOLDS
    print(f"Total runs      : {total_runs}")
    print("=" * 72)

    all_rows = []
    run_idx = 0
    grand_t0 = time.time()

    for dataset in datasets:
        for method_name in METHODS:
            for bins in BINS_LIST:
                for fold in range(1, N_FOLDS + 1):
                    run_idx += 1
                    combo = combo_string(method_name, bins)
                    print(f"\n[{run_idx}/{total_runs}] "
                          f"{dataset['name']} | {combo} | fold {fold}")
                    try:
                        rows = run_single_fold(dataset, method_name, bins, fold)
                        all_rows.extend(rows)
                        print(f"  -> {len(rows)} cutoff rows collected")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        import traceback
                        traceback.print_exc()

    if not all_rows:
        print("\nNo cutoff rows produced. Check dataset paths.")
        return

    wide = pivot_to_wide(all_rows)
    wide.to_csv(output_csv, index=False)

    grand_elapsed = time.time() - grand_t0
    print("\n" + "=" * 72)
    print(f"Wrote {len(wide)} rows to {output_csv}")
    print(f"Total wall time: {grand_elapsed / 60:.1f} min")

    print("\nRows per (dataset, combo):")
    counts = wide.groupby(["dataset", "combo"]).size().reset_index(name="rows")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
