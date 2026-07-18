"""
Stage 5 (MARIO forecasting): cross-TIRP aggregation + forecast evaluation.

This is the MARIO rewrite of the old FCPM ROC/TTE aggregation. It consumes the
per-TIRP forecasts written by Stage 4 (`tirp_<id>/forecasts.csv.gz`, each row a
`(EntityID, current_time)` with a `P_<symbol>` probability distribution) and
collapses them into ONE forecast per `(entity, t)`:

  1. Active set. A TIRP is "active" at timestamp t for an entity if it produced a
     forecast row there. Stage 4 emits a row for every integer t a TIRP prefix
     instance spans, so "has a row at t" == "prefix active at t". With
     `context_window = C > 0` a TIRP also counts as active at t if its most recent
     row falls in [t - C, t] (a grace period after the pattern completes); its
     contribution is then that most-recent in-window distribution.
  2. Symbol alignment. Each Stage 3 model only predicts the target symbols it saw
     in training, so different TIRPs have different `P_<symbol>` columns. Every
     TIRP's distribution is reindexed to the global union of symbols, filling
     missing symbols with 0.0 (the model assigns them zero probability). Each row
     therefore still sums to 1.
  3. Aggregation. The active TIRPs' distributions are combined per symbol
     (default: unweighted average) and the argmax is the forecast symbol.
  4. Evaluation. Only rows with valid ground truth (`covered`) are scored, and
     only those inside the test region per the Stage 0 split manifest:
     `seen_future` entities are scored where `current_time > cut_time`;
     `new_entity` (holdout) entities are scored across their timeline (after an
     optional per-entity `warmup`). Multiclass accuracy / macro-F1 / weighted-F1 /
     log-loss are reported overall and per regime against a majority baseline.

Outputs (into results_output_dir):
    aggregated_forecasts.csv.gz   # one row per (entity, t): aggregated P_<symbol>,
                                  #   pred_symbol, y_true, covered, regime, scored
    stage5_metrics.csv            # one row per scope (overall / seen_future / new_entity)
    stage5_aggregation.done       # resumability marker
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


# --- Loading -----------------------------------------------------------------

def load_all_forecasts(prediction_output_dir):
    """
    Read every `tirp_<id>/forecasts.csv.gz` under prediction_output_dir.

    Returns (forecasts, symbols):
      forecasts : long DataFrame [tirp_id, EntityID, current_time, y_true,
                  covered, P_<symbol>...] with each TIRP's P-columns reindexed to
                  the global symbol union (missing symbols filled 0.0).
      symbols   : sorted list[int] of all target symbols seen across TIRPs.
    A TIRP whose Stage 4 run produced no rows (empty forecasts) is skipped.
    """
    paths = sorted(glob.glob(os.path.join(prediction_output_dir, "tirp_*", "forecasts.csv.gz")))
    if not paths:
        print(f"ERROR: No 'tirp_*/forecasts.csv.gz' found under {prediction_output_dir}.")
        sys.exit(1)

    frames = []
    symbols = set()
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        tirp_id = os.path.basename(os.path.dirname(p))[len("tirp_"):]
        df["tirp_id"] = tirp_id
        frames.append(df)
        symbols.update(int(c[len("P_"):]) for c in df.columns if c.startswith("P_"))

    if not frames:
        print(f"ERROR: All forecast files under {prediction_output_dir} were empty.")
        sys.exit(1)

    symbols = sorted(symbols)
    prob_cols = [f"P_{s}" for s in symbols]
    meta_cols = ["tirp_id", "EntityID", "current_time", "y_true", "covered"]

    aligned = []
    for df in frames:
        # Reindex to the global symbol set; symbols this model never saw -> 0 prob.
        for c in prob_cols:
            if c not in df.columns:
                df[c] = 0.0
        aligned.append(df[meta_cols + prob_cols])

    forecasts = pd.concat(aligned, ignore_index=True)
    forecasts["EntityID"] = forecasts["EntityID"].astype(int)
    forecasts["current_time"] = forecasts["current_time"].astype(int)
    forecasts["y_true"] = forecasts["y_true"].astype(int)
    forecasts["covered"] = forecasts["covered"].astype(bool)
    print(f"Loaded {len(paths)} TIRP forecast file(s), {len(forecasts)} rows, "
          f"{len(symbols)} target symbols {symbols}.")
    return forecasts, symbols


# --- Aggregation -------------------------------------------------------------

def aggregate_forecasts(forecasts, symbols, context_window=0, method="average"):
    """
    Collapse per-TIRP forecasts into one distribution per (EntityID, current_time).

    Steps:
      * Reduce overlapping instances of the SAME TIRP at the same (entity, t) to a
        single distribution (mean), so each TIRP contributes at most one vote per t.
      * Combine the active TIRPs' distributions per symbol:
          - 'average': unweighted mean across active TIRPs.
          - 'max':     per-symbol maximum across active TIRPs (then renormalised).
        where "active at t" is:
          - context_window == 0: all TIRPs with a row exactly at t (== all TIRPs
            whose prefix spans t).
          - context_window  > 0: for each evaluation point t (drawn from the union
            of exact rows) and each TIRP, that TIRP's most-recent row with
            current_time in [t - context_window, t] (merge_asof backward).
      * argmax over the aggregated distribution -> pred_symbol.

    Returns a DataFrame [EntityID, current_time, y_true, covered, n_active_tirps,
    P_<symbol>..., pred_symbol].
    """
    if method not in ("average", "max"):
        raise ValueError(f"Unsupported aggregation method '{method}' (expected 'average' or 'max').")

    prob_cols = [f"P_{s}" for s in symbols]

    # Per-TIRP reduction: mean distribution when a TIRP has several instances at t.
    reduced = (
        forecasts.groupby(["tirp_id", "EntityID", "current_time"], as_index=False)[prob_cols].mean()
    )

    # Ground truth is a property of (EntityID, current_time), identical across TIRPs.
    truth = (
        forecasts.groupby(["EntityID", "current_time"], as_index=False)
        .agg(y_true=("y_true", "first"), covered=("covered", "first"))
    )

    if context_window == 0:
        grp = reduced.groupby(["EntityID", "current_time"])
        combined = grp[prob_cols].mean() if method == "average" else grp[prob_cols].max()
        agg = combined.reset_index()
        agg["n_active_tirps"] = grp.size().values
    else:
        # Evaluation grid = every (entity, t) that any TIRP forecast exactly.
        # merge_asof needs the `on` key (current_time) globally sorted, so we sort
        # by current_time (EntityID is handled via `by=`) and keep this row order
        # as canonical for the accumulator arrays below.
        grid = (
            reduced[["EntityID", "current_time"]]
            .drop_duplicates()
            .sort_values(["current_time", "EntityID"])
            .reset_index(drop=True)
        )
        acc = np.zeros((len(grid), len(prob_cols)), dtype=np.float64)
        active = np.zeros(len(grid), dtype=np.int64)
        for _, right in reduced.groupby("tirp_id"):
            right = right.sort_values(["current_time", "EntityID"])
            m = pd.merge_asof(
                grid, right[["EntityID", "current_time"] + prob_cols],
                by="EntityID", on="current_time",
                direction="backward", tolerance=int(context_window),
            )
            vals = m[prob_cols].to_numpy(dtype=np.float64)
            hit = ~np.isnan(vals).any(axis=1)   # this TIRP had an in-window row
            if method == "average":
                acc[hit] += vals[hit]
            else:  # max: probabilities are >= 0, so 0-init is a valid lower bound
                acc[hit] = np.maximum(acc[hit], vals[hit])
            active += hit
        agg = grid.copy()
        agg[prob_cols] = acc / active[:, None] if method == "average" else acc  # active >= 1
        agg["n_active_tirps"] = active

    # Renormalize each aggregated row to sum to exactly 1. For 'average' this only
    # removes the ~1e-7 float drift XGBoost's softprob carries (which otherwise
    # trips log_loss); for 'max' it is a real renormalisation into a distribution.
    row_sums = agg[prob_cols].sum(axis=1)
    agg[prob_cols] = agg[prob_cols].div(row_sums, axis=0)

    agg = agg.merge(truth, on=["EntityID", "current_time"], how="left")
    agg["pred_symbol"] = np.asarray(symbols)[agg[prob_cols].to_numpy().argmax(axis=1)]

    ordered = ["EntityID", "current_time", "y_true", "covered", "n_active_tirps"] + prob_cols + ["pred_symbol"]
    return agg[ordered].sort_values(["EntityID", "current_time"]).reset_index(drop=True)


# --- Regime filtering --------------------------------------------------------

def attach_regime_and_filter(agg, manifest, warmup=0):
    """
    Attach the Stage 0 test regime to each aggregated row and flag which rows are
    scored. Adds columns: test_regime, cut_time, scored.

      * seen_future (train entities): scored where current_time > cut_time -- only
        the revealed-future slice, never the training history.
      * new_entity (holdout entities): scored across the timeline once past a
        per-entity warm-up (current_time >= first_forecast_time + warmup).

    Rows whose entity is absent from the manifest are marked scored=False.
    """
    m = manifest[["EntityID", "test_regime", "cut_time"]].copy()
    m["EntityID"] = m["EntityID"].astype(int)
    out = agg.merge(m, on="EntityID", how="left")

    first_time = out.groupby("EntityID")["current_time"].transform("min")
    is_seen = out["test_regime"] == "seen_future"
    is_new = out["test_regime"] == "new_entity"

    scored = pd.Series(False, index=out.index)
    scored |= is_seen & out["cut_time"].notna() & (out["current_time"] > out["cut_time"])
    scored |= is_new & (out["current_time"] >= first_time + warmup)
    out["scored"] = scored & out["covered"].astype(bool)
    return out


# --- Evaluation --------------------------------------------------------------

def _metrics_for(subset, symbols):
    """Multiclass metrics + majority baseline for one scored subset."""
    prob_cols = [f"P_{s}" for s in symbols]
    y = subset["y_true"].to_numpy()
    pred = subset["pred_symbol"].to_numpy()
    n = len(subset)

    counts = subset["y_true"].value_counts()
    majority_class = int(counts.index[0])
    majority_baseline_acc = float(counts.iloc[0] / n)

    # log-loss needs y_true to be among the probability columns' symbols.
    in_vocab = subset["y_true"].isin(symbols)
    if in_vocab.any():
        ll = float(log_loss(
            subset.loc[in_vocab, "y_true"].to_numpy(),
            subset.loc[in_vocab, prob_cols].to_numpy(),
            labels=symbols,
        ))
    else:
        ll = float("nan")

    return {
        "n": n,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=symbols, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=symbols, zero_division=0)),
        "log_loss": ll,
        "majority_class": majority_class,
        "majority_baseline_acc": majority_baseline_acc,
        "beats_baseline": bool(accuracy_score(y, pred) > majority_baseline_acc),
    }


def evaluate(scored_agg, symbols):
    """
    Metrics for the scored rows overall and per regime.
    Returns a DataFrame, one row per scope.
    """
    scored = scored_agg[scored_agg["scored"]]
    rows = []
    if len(scored):
        rows.append({"scope": "overall", **_metrics_for(scored, symbols)})
    for regime in ["seen_future", "new_entity"]:
        sub = scored[scored["test_regime"] == regime]
        if len(sub):
            rows.append({"scope": regime, **_metrics_for(sub, symbols)})
    if not rows:
        print("WARNING: no scored rows to evaluate (check manifest / cut_time / warmup).")
    return pd.DataFrame(rows)


# --- Orchestration -----------------------------------------------------------

def run_single_aggregation_evaluation(prediction_output_dir, split_manifest_path,
                                      results_output_dir, aggregation_method="average",
                                      context_window=0, warmup=0):
    print("--- Starting Stage 5: Aggregation & Forecast Evaluation (MARIO) ---")
    print(f"Predictions dir : {prediction_output_dir}")
    print(f"Split manifest  : {split_manifest_path}")
    print(f"Results dir     : {results_output_dir}")
    print(f"Aggregation     : {aggregation_method} | context_window={context_window} | warmup={warmup}")
    t0 = time.time()

    if not os.path.exists(split_manifest_path):
        print(f"ERROR: Split manifest not found at {split_manifest_path}")
        sys.exit(1)
    manifest = pd.read_csv(split_manifest_path)

    forecasts, symbols = load_all_forecasts(prediction_output_dir)
    agg = aggregate_forecasts(forecasts, symbols, context_window=context_window,
                              method=aggregation_method)
    agg = attach_regime_and_filter(agg, manifest, warmup=warmup)
    metrics = evaluate(agg, symbols)

    os.makedirs(results_output_dir, exist_ok=True)
    agg_path = os.path.join(results_output_dir, "aggregated_forecasts.csv.gz")
    agg.to_csv(agg_path, index=False, compression="gzip")
    metrics_path = os.path.join(results_output_dir, "stage5_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print(f"\nAggregated {len(agg)} (entity, t) forecasts -> {agg_path}")
    print(f"Scored {int(agg['scored'].sum())} rows. Metrics -> {metrics_path}")
    if len(metrics):
        print(metrics.to_string(index=False))

    done_path = os.path.join(results_output_dir, "stage5_aggregation.done")
    with open(done_path, "w") as f:
        f.write("done")
    print(f"\n--- Finished Stage 5 in {time.time() - t0:.2f}s ---")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5 (MARIO): cross-TIRP forecast aggregation + evaluation."
    )
    parser.add_argument("--prediction_output_dir", required=True,
                        help="Dir holding Stage 4 'tirp_<id>/forecasts.csv.gz' files.")
    parser.add_argument("--split_manifest_path", required=True,
                        help="Stage 0 'split_manifest.csv' (EntityID, role, test_regime, cut_time).")
    parser.add_argument("--results_output_dir", required=True,
                        help="Dir to write aggregated_forecasts.csv.gz + stage5_metrics.csv.")
    parser.add_argument("--aggregation_method", default="average",
                        help="Cross-TIRP aggregation function: 'average' or 'max'.")
    parser.add_argument("--context_window", type=int, default=0,
                        help="Grace window C: a TIRP is active at t if its most recent "
                             "forecast falls in [t-C, t]. 0 = exact (prefix spans t).")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Per-entity warm-up (time units) before new_entity rows are scored.")
    args = parser.parse_args()

    run_single_aggregation_evaluation(
        prediction_output_dir=args.prediction_output_dir,
        split_manifest_path=args.split_manifest_path,
        results_output_dir=args.results_output_dir,
        aggregation_method=args.aggregation_method,
        context_window=args.context_window,
        warmup=args.warmup,
    )
    sys.exit(0)
