import os
import argparse
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import logging
import gc
import traceback
import hashlib

# MARIO: rebuild the exact training matrix Stage 3 trained on (durations already saved
# per TIRP; re-derive the t+HORIZON labels) so we can score the model back on it.
# build_forecast_inference_matrix is the row-preserving twin (keeps per-(entity, t)
# metadata) used to build the forecast-vs-truth timeline for visualisation.
from CPM_Feature_Matrix.Create_feature_matrix import (
    build_forecast_training_arrays, build_forecast_inference_matrix,
)
# Shared metric core: classification (accuracy / macro-F1 / weighted-F1 / log-loss vs a
# majority baseline) + ordinal regression (MSE / MAE / RMSE via the abstraction bin
# centres). Same code the eda notebook and the whole-experiment sweeps use, so the
# per-TIRP train numbers here stay identical to those.
from tirp_forecast_eval import forecast_metrics, build_state_value_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_sanitized_tirp_id(tirp_object_file_path):
    """ Same sanitization logic as in Stage 3. """
    try:
        tirp_filename = os.path.basename(tirp_object_file_path)
        tirp_id_base = os.path.splitext(tirp_filename)[0]
        sanitized_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in tirp_id_base)
        if not sanitized_id:
            hasher = hashlib.sha1(tirp_object_file_path.encode())
            sanitized_id = f"tirp_hash_{hasher.hexdigest()[:8]}"
        return sanitized_id
    except Exception as e:
        path_hash = hashlib.sha1(tirp_object_file_path.encode()).hexdigest()[:8]
        return f"tirp_error_id_{path_hash}"


def load_target_stis(abstraction_output_dir, target_variable):
    """
    Load the forecast target variable's STIs from the training KL (same source Stage 3
    labels against). Returns DataFrame [EntityID, StartTime, EndTime, StateID].
    """
    # Reuse Stage 3's KL parser to stay in lockstep with how labels were built.
    from run_stage3_build_model import txt_2_csv
    train_kl_path = os.path.join(abstraction_output_dir, 'Train', 'KL.txt')
    all_stis = txt_2_csv(train_kl_path)
    target_stis = all_stis[all_stis["TemporalPropertyID"] == int(target_variable)][
        ["EntityID", "StartTime", "EndTime", "StateID"]
    ].copy()
    target_stis["EntityID"] = target_stis["EntityID"].astype(int)
    return target_stis


def run_train_validation_for_tirp(tirp_path, tirp_model_run_dir, target_stis, horizon,
                                  state_to_value=None):
    """
    MARIO Stage 3.5 -- a train-set forecast-accuracy sanity check for one TIRP's model.

    Rebuilds the exact per-(entity, t) matrix Stage 3 trained on (from the saved
    durations table + the target STIs, labelled at t + horizon), scores the saved
    multiclass forecast model on it, and writes multiclass metrics
    (accuracy / macro-F1 / weighted-F1 / log-loss) alongside a majority-class baseline
    so an uninformative model is obvious. When ``state_to_value`` is given (a StateID ->
    real bin-centre map), ordinal-regression metrics (MSE / MAE / RMSE) are added too.
    This is a train-set check, NOT Stage 5 (which aggregates across TIRPs on the test set).

    Resumable via a stage3_val_<id>.done marker.
    """
    sanitized_id = get_sanitized_tirp_id(tirp_path)
    tirp_output_dir = os.path.join(tirp_model_run_dir, f'tirp_{sanitized_id}')

    done_file_path = os.path.join(tirp_output_dir, f'stage3_val_{sanitized_id}.done')
    if os.path.exists(done_file_path):
        logging.info(f"Skipping TIRP {sanitized_id} (Already Validated: {done_file_path})")
        return True

    logging.info(f"Starting MARIO train validation for TIRP: {sanitized_id}")

    # 1. Reload the durations table Stage 3 saved (one row per detected instance).
    duration_file = os.path.join(tirp_output_dir, "durations_merged_df.csv")
    if not os.path.exists(duration_file):
        logging.warning(f"Durations table missing for {sanitized_id}: {duration_file}. Cannot validate.")
        return False

    # 2. Locate the trained forecast model (exactly one CPML pkl per TIRP dir).
    model_matches = glob.glob(os.path.join(tirp_output_dir, 'models', '*-CPML.pkl'))
    if not model_matches:
        logging.warning(f"No trained model for {sanitized_id} under models/. Was Stage 3 run?")
        return False
    model_path = model_matches[0]

    try:
        durations_df = pd.read_csv(duration_file, low_memory=False)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 3. Rebuild the labelled training matrix (identical to Stage 3).
        X, y_true, _ = build_forecast_training_arrays(durations_df, target_stis, horizon)

        n_rows = int(X.shape[0])
        if n_rows == 0:
            logging.warning(f"TIRP {sanitized_id}: 0 labelled rows; writing empty summary.")
            m = forecast_metrics(np.array([]), np.array([]), state_to_value=state_to_value)
        else:
            # 4. Score on train. proba columns align to model.classes_ (the real symbols).
            proba = model.predict_proba_matrix(X)
            classes = np.asarray(model.classes_)
            y_pred = classes[proba.argmax(axis=1)]
            # Single shared core: classification + (if a state->value map is given)
            # ordinal-regression metrics, next to the majority-class baseline.
            m = forecast_metrics(y_true, y_pred, proba=proba, classes=classes,
                                 state_to_value=state_to_value)

        # Map the generic metric keys onto Stage 3.5's train_* column schema (kept stable
        # for any downstream reader), then append the regression columns.
        summary = {
            "TIRP_name": sanitized_id,
            "n_rows": m["n_rows"],
            "n_classes": m["n_classes"],
            "train_accuracy": m["accuracy"],
            "train_macro_f1": m["macro_f1"],
            "train_weighted_f1": m["weighted_f1"],
            "train_logloss": m["logloss"],
            "majority_class": m["majority_class"],
            "majority_baseline_acc": m["majority_baseline_acc"],
            "train_mse": m["mse"],
            "train_mae": m["mae"],
            "train_rmse": m["rmse"],
        }
        if n_rows:
            logging.info(f"{sanitized_id}: acc={summary['train_accuracy']:.3f} "
                         f"(majority {summary['majority_baseline_acc']:.3f}), "
                         f"macroF1={summary['train_macro_f1']:.3f}, "
                         f"logloss={summary['train_logloss']:.3f}, "
                         f"mae={summary['train_mae']!s:.5}")

        summary_file_path = os.path.join(tirp_output_dir, "train_summary_metrics.csv")
        pd.DataFrame([summary]).to_csv(summary_file_path, index=False)

        with open(done_file_path, 'w') as f_done:
            f_done.write("done")

    except Exception as e:
        logging.error(f"Error during validation for {sanitized_id}: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()

    return True


def build_forecast_timeline_for_tirp(tirp_path, tirp_model_run_dir, target_stis, horizon):
    """
    Per-(entity, current_time) forecast-vs-truth timeline for ONE built model, on the
    TRAIN set -- the row-by-row data behind the Stage 3.5 accuracy number, kept so it
    can be plotted (the forecast made at each current time t, for t + horizon, against
    the true symbol there).

    Uses ``build_forecast_inference_matrix`` (not the training builder) so no rows are
    dropped and every prediction carries its (EntityID, current_time) -- the X features
    are identical to training, so predictions match the Stage 3.5 metrics.

    A TIRP can have several prefix instances active at the same (entity, current_time)
    -- one per occurrence of the pattern -- each yielding its own prediction row. These
    are collapsed to ONE forecast per (entity, current_time) by averaging their
    probability distributions then taking the argmax (the same per-TIRP reduction Stage 5
    uses), so a timestamp has a single forecast rather than a stack of them. ``n_instances``
    records how many instances backed each point.

    Returns DataFrame [EntityID, current_time, target_time, y_true, y_pred, covered,
    correct, n_instances] (target_time = current_time + horizon; y_true is meaningful only
    where covered=True). Returns None if the durations table or model is missing.
    """
    sanitized_id = get_sanitized_tirp_id(tirp_path)
    tirp_output_dir = os.path.join(tirp_model_run_dir, f'tirp_{sanitized_id}')
    duration_file = os.path.join(tirp_output_dir, "durations_merged_df.csv")
    model_matches = glob.glob(os.path.join(tirp_output_dir, 'models', '*-CPML.pkl'))
    cols = ["EntityID", "current_time", "target_time", "y_true", "y_pred", "covered", "correct", "n_instances"]

    if not os.path.exists(duration_file) or not model_matches:
        logging.warning(f"Timeline unavailable for {sanitized_id}: missing durations or model.")
        return None

    durations_df = pd.read_csv(duration_file, low_memory=False)
    with open(model_matches[0], 'rb') as f:
        model = pickle.load(f)

    X, meta, _ = build_forecast_inference_matrix(durations_df, target_stis, horizon)
    if X.shape[0] == 0:
        return pd.DataFrame(columns=cols)

    classes = np.asarray(model.classes_)
    proba = model.predict_proba_matrix(X)                      # (n_rows, n_classes)
    prob_cols = [f"_p{j}" for j in range(len(classes))]

    df = meta.reset_index(drop=True).copy()
    df[prob_cols] = proba

    # Collapse overlapping instances: mean probability per (entity, t), then argmax.
    grp = df.groupby(["EntityID", "current_time"])
    out = grp.agg(y_true=("y_true", "first"), covered=("covered", "first"),
                  n_instances=("y_true", "size")).reset_index()
    mean_proba = grp[prob_cols].mean().to_numpy()
    out["y_pred"] = classes[mean_proba.argmax(axis=1)]
    out["target_time"] = out["current_time"] + horizon
    out["covered"] = out["covered"].astype(bool)
    out["correct"] = out["covered"] & (out["y_pred"] == out["y_true"])
    return out[cols].sort_values(["EntityID", "current_time"]).reset_index(drop=True)


def plot_forecast_timeline(timeline_df, entity_id, horizon, focus_t=None, ax=None,
                           title=None, max_points=None, mark_all_current=False):
    """
    Plot one entity's forecast-vs-truth timeline for a single model.

    x-axis = global timestamp, y-axis = target symbol id. For each inference the model
    makes while standing at current time t, the forecast is for t + horizon:
      * green  = the true target symbol at t + horizon
      * blue   = the model's predicted symbol at t + horizon
      * red    = the current time t the model sees (a solid vertical line); the dotted
                 red line + shaded band mark the horizon lead up to t + horizon.
    Only ``covered`` rows (a real symbol exists at t + horizon) are plotted.

    ``focus_t`` chooses which inference step gets the red cursor (default: the middle
    covered prediction). ``mark_all_current=True`` additionally draws a faint red line at
    every inference origin. Returns the matplotlib Axes. matplotlib is imported lazily so
    the SLURM entry point never depends on it.
    """
    import matplotlib.pyplot as plt

    df = timeline_df[(timeline_df["EntityID"] == entity_id) & timeline_df["covered"]].copy()
    df = df.sort_values("current_time")
    if max_points and len(df) > max_points:
        df = df.iloc[:: max(1, len(df) // max_points)]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.6))
    if df.empty:
        ax.set_title(title or f"Entity {entity_id}: no covered forecasts")
        return ax

    acc = float((df["y_pred"] == df["y_true"]).mean())

    # True vs predicted symbol at the forecast target time (t + horizon).
    ax.plot(df["target_time"], df["y_true"], color="green", marker="o", ms=5, lw=1.0,
            alpha=0.7, label="true symbol at t+H", zorder=2)
    ax.scatter(df["target_time"], df["y_pred"], color="tab:blue", marker="x", s=45, lw=1.6,
               label="predicted symbol at t+H", zorder=3)

    if mark_all_current:
        for t in df["current_time"]:
            ax.axvline(t, color="red", lw=0.5, alpha=0.12)

    # Red cursor: one representative "current time t" and its horizon lead to t + horizon.
    if focus_t is None:
        focus_t = int(df["current_time"].iloc[len(df) // 2])
    target_t = focus_t + horizon
    ax.axvline(focus_t, color="red", lw=2.0, label=f"current time t = {focus_t}", zorder=4)
    ax.axvline(target_t, color="red", ls=":", lw=1.2, alpha=0.7, zorder=4)
    ax.axvspan(focus_t, target_t, color="red", alpha=0.06, zorder=0)
    focus_row = df[df["current_time"] == focus_t]
    if len(focus_row):
        r = focus_row.iloc[0]
        ax.scatter([target_t], [r["y_true"]], facecolors="none", edgecolors="green", s=160, lw=1.8, zorder=5)
        ax.scatter([target_t], [r["y_pred"]], facecolors="none", edgecolors="tab:blue", s=160, lw=1.8, zorder=5)

    syms = sorted(set(df["y_true"]).union(df["y_pred"]))
    ax.set_yticks(syms)
    ax.set_xlabel("global timestamp")
    ax.set_ylabel("target symbol id")
    ax.set_title(title or f"Entity {entity_id}: forecast vs. true (horizon={horizon}, acc={acc:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARIO Stage 3.5: train-set forecast-accuracy check per TIRP.")

    parser.add_argument("--tirp_model_run_dir", required=True, help="Base directory (feature_matrix dir).")
    parser.add_argument("--tirp_list_file", required=True, help="Text file listing TIRP .pkl paths.")
    parser.add_argument("--abstraction_output_dir", required=True, help="Stage 1 output dir (for Train/KL.txt).")
    parser.add_argument("--target_variable", required=True, type=int,
                        help="TemporalPropertyID whose future state is forecast (must match Stage 3).")
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon (must match Stage 3).")
    parser.add_argument("--data_path", default=None,
                        help="Optional raw long data CSV (EntityID, TemporalPropertyID, "
                             "TimeStamp, TemporalPropertyValue). Used only to place the open "
                             "tail bins when mapping target StateIDs to real values for the "
                             "MSE/MAE/RMSE metrics. If omitted, the tails are closed from the "
                             "states.csv bin widths.")

    args = parser.parse_args()

    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)

    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]

    logging.info(f"Found {len(tirp_paths)} TIRPs for validation.")
    target_stis = load_target_stis(args.abstraction_output_dir, args.target_variable)

    # Build the StateID -> real bin-centre map for the regression metrics from the Train
    # states.csv (auto-located next to Train/KL.txt). Optional: never fail Stage 3.5 if the
    # states file is unusable -- just skip the MSE/MAE/RMSE columns in that case.
    state_to_value = None
    train_states_path = os.path.join(args.abstraction_output_dir, "Train", "states.csv")
    try:
        state_to_value = build_state_value_map(
            train_states_path, target_variable=args.target_variable, data=args.data_path,
        )
    except Exception as e:
        logging.warning(f"Could not build state->value map from {train_states_path} "
                        f"({e}); MSE/MAE/RMSE will be NaN.")

    tirp_val_success = True
    for tirp_path in tirp_paths:
        success = run_train_validation_for_tirp(
            tirp_path=tirp_path,
            tirp_model_run_dir=args.tirp_model_run_dir,
            target_stis=target_stis,
            horizon=args.horizon,
            state_to_value=state_to_value,
        )
        if not success:
            tirp_val_success = False

    if tirp_val_success:
        logging.info("Stage 3.5 completed successfully for all TIRPs in the batch.")
        sys.exit(0)
    else:
        logging.error("Stage 3.5 failed for at least one TIRP in the batch.")
        sys.exit(1)
