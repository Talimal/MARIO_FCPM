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

from sklearn.metrics import accuracy_score, f1_score, log_loss

# MARIO: rebuild the exact training matrix Stage 3 trained on (durations already saved
# per TIRP; re-derive the t+HORIZON labels) so we can score the model back on it.
from CPM_Feature_Matrix.Create_feature_matrix import build_forecast_training_arrays

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


def run_train_validation_for_tirp(tirp_path, tirp_model_run_dir, target_stis, horizon):
    """
    MARIO Stage 3.5 -- a train-set forecast-accuracy sanity check for one TIRP's model.

    Rebuilds the exact per-(entity, t) matrix Stage 3 trained on (from the saved
    durations table + the target STIs, labelled at t + horizon), scores the saved
    multiclass forecast model on it, and writes multiclass metrics
    (accuracy / macro-F1 / weighted-F1 / log-loss) alongside a majority-class baseline
    so an uninformative model is obvious. This is a train-set check, NOT Stage 5
    (which aggregates across TIRPs on the test set).

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
            summary = {"TIRP_name": sanitized_id, "n_rows": 0, "n_classes": 0,
                       "train_accuracy": np.nan, "train_macro_f1": np.nan,
                       "train_weighted_f1": np.nan, "train_logloss": np.nan,
                       "majority_class": np.nan, "majority_baseline_acc": np.nan}
        else:
            # 4. Score on train. proba columns align to model.classes_ (the real symbols).
            proba = model.predict_proba_matrix(X)
            classes = np.asarray(model.classes_)
            y_pred = classes[proba.argmax(axis=1)]

            # Majority-class baseline: the accuracy of always predicting the commonest
            # training symbol -- the bar the model must clear to be informative.
            vals, counts = np.unique(y_true, return_counts=True)
            majority_class = vals[counts.argmax()]
            majority_baseline_acc = counts.max() / n_rows

            summary = {
                "TIRP_name": sanitized_id,
                "n_rows": n_rows,
                "n_classes": int(len(classes)),
                "train_accuracy": accuracy_score(y_true, y_pred),
                "train_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "train_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "train_logloss": log_loss(y_true, proba, labels=classes),
                "majority_class": int(majority_class),
                "majority_baseline_acc": majority_baseline_acc,
            }
            logging.info(f"{sanitized_id}: acc={summary['train_accuracy']:.3f} "
                         f"(majority {summary['majority_baseline_acc']:.3f}), "
                         f"macroF1={summary['train_macro_f1']:.3f}, "
                         f"logloss={summary['train_logloss']:.3f}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARIO Stage 3.5: train-set forecast-accuracy check per TIRP.")

    parser.add_argument("--tirp_model_run_dir", required=True, help="Base directory (feature_matrix dir).")
    parser.add_argument("--tirp_list_file", required=True, help="Text file listing TIRP .pkl paths.")
    parser.add_argument("--abstraction_output_dir", required=True, help="Stage 1 output dir (for Train/KL.txt).")
    parser.add_argument("--target_variable", required=True, type=int,
                        help="TemporalPropertyID whose future state is forecast (must match Stage 3).")
    parser.add_argument("--horizon", required=True, type=int, help="Forecast horizon (must match Stage 3).")

    args = parser.parse_args()

    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)

    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]

    logging.info(f"Found {len(tirp_paths)} TIRPs for validation.")
    target_stis = load_target_stis(args.abstraction_output_dir, args.target_variable)

    tirp_val_success = True
    for tirp_path in tirp_paths:
        success = run_train_validation_for_tirp(
            tirp_path=tirp_path,
            tirp_model_run_dir=args.tirp_model_run_dir,
            target_stis=target_stis,
            horizon=args.horizon,
        )
        if not success:
            tirp_val_success = False

    if tirp_val_success:
        logging.info("Stage 3.5 completed successfully for all TIRPs in the batch.")
        sys.exit(0)
    else:
        logging.error("Stage 3.5 failed for at least one TIRP in the batch.")
        sys.exit(1)
