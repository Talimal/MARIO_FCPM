import os
import argparse
import sys
import pickle
import glob
import time
import gc
import traceback

import numpy as np
import pandas as pd

from config import *

# --- MARIO Stage 4 imports ---
# Detect each TIRP's evolving prefixes on the TEST KL (event-free), expand to a
# per-(entity, absolute t) matrix WITH row metadata, then score the Stage 3 model.
from CPM_Feature_Matrix.Create_feature_matrix import (
    build_forecast_durations,
    build_forecast_inference_matrix,
)
# Reuse Stage 3's KL parser + id sanitizer so Stage 4 stays in lockstep with how the
# models were built (same TIRP ids, same target-STI extraction).
from run_stage3_build_model import txt_2_csv, get_sanitized_tirp_id
from FCPM_Package.CPML import CPML  # noqa: F401  (needed to unpickle saved models)


# --- Batch-level common data -----------------------------------------------------

def prepare_batch_data(abstraction_output_dir, target_variable):
    """
    Common data shared by every TIRP in a Stage 4 batch: the path to the TEST KL
    (where each TIRP's prefixes are detected on unseen data) and the forecast target
    variable's STIs from that same TEST KL (the ground-truth symbol at t + HORIZON used
    to evaluate the forecasts).

    Mirrors Stage 3's ``prepare_batch_data`` but reads Test/KL.txt instead of Train/KL.txt.

    :param abstraction_output_dir: Stage 1 output dir (expects Test/KL.txt).
    :param target_variable: TemporalPropertyID whose future abstracted state is forecast.
    :return: dict with 'test_kl_path' and 'target_stis' (DataFrame
             [EntityID, StartTime, EndTime, StateID], filtered to the target variable).
    """
    test_kl_path = os.path.join(abstraction_output_dir, 'Test', 'KL.txt')
    if not os.path.exists(test_kl_path):
        raise FileNotFoundError(f"Test KL not found: {test_kl_path}")

    print(f"--- Preparing batch data: loading target STIs (TPID={target_variable}) from {test_kl_path} ---")
    all_stis = txt_2_csv(test_kl_path)
    target_stis = all_stis[all_stis["TemporalPropertyID"] == int(target_variable)][
        ["EntityID", "StartTime", "EndTime", "StateID"]
    ].copy()
    target_stis["EntityID"] = target_stis["EntityID"].astype(int)

    n_ent = target_stis["EntityID"].nunique()
    n_states = target_stis["StateID"].nunique()
    print(f"Target variable {target_variable}: {len(target_stis)} test STIs over {n_ent} entities, "
          f"{n_states} distinct states.")
    if target_stis.empty:
        print(f"WARNING: no test STIs for target variable {target_variable} -- forecasts will have "
              f"no ground truth (all rows covered=False).")

    return {"test_kl_path": test_kl_path, "target_stis": target_stis}


def find_tirp_model(built_models_base_dir, sanitized_id):
    """ Locate the Stage 3 model for a TIRP (exactly one CPML pkl per tirp_<id>/models). """
    matches = glob.glob(
        os.path.join(built_models_base_dir, f'tirp_{sanitized_id}', 'models', '*-CPML.pkl')
    )
    return matches[0] if matches else None


# --- Per-TIRP forecast inference -------------------------------------------------

def process_single_tirp(tirp_path, built_models_base_dir, prediction_output_dir,
                        common_data, max_gap, num_relations, epsilon, horizon):
    """
    Run MARIO forecast inference for one TIRP on the TEST set:
      1. detect the TIRP's evolving prefixes on the TEST KL and extract per-instance
         TIEP durations (build_forecast_durations, event-free);
      2. expand to a per-(entity, absolute t) matrix WITH metadata + ground-truth label
         at t + horizon (build_forecast_inference_matrix, no rows dropped);
      3. load the Stage 3 model and predict the full symbol distribution per row
         (predict_proba_matrix), aligned to the model's classes_;
      4. save one forecasts table for this TIRP (EntityID, current_time, TFS, y_true,
         covered, and one probability column per target symbol).

    Same batch-by-TIRP shape and .done resumability as Stage 3.
    """
    debug_params = {}
    try:
        sanitized_id = get_sanitized_tirp_id(tirp_path)
        current_tirp_dir = os.path.join(prediction_output_dir, f'tirp_{sanitized_id}')
        debug_params = {
            "tirp_path": tirp_path, "sanitized_id": sanitized_id,
            "current_tirp_dir": current_tirp_dir, "max_gap": max_gap,
            "num_relations": num_relations, "epsilon": epsilon, "horizon": horizon,
        }

        print(f"\n--- Forecasting TIRP: {sanitized_id} ---")
        done_file_path = os.path.join(current_tirp_dir, f'stage4_predict_{sanitized_id}.done')
        if os.path.exists(done_file_path):
            print(f"Skipping TIRP {sanitized_id} (Already Done: {done_file_path})")
            return True
        os.makedirs(current_tirp_dir, exist_ok=True)

        # --- Step 0: locate the Stage 3 model (no model -> nothing to forecast) ---
        model_path = find_tirp_model(built_models_base_dir, sanitized_id)
        if model_path is None:
            print(f"TIRP {sanitized_id}: no Stage 3 model under "
                  f"{built_models_base_dir}/tirp_{sanitized_id}/models/. Marking done, no forecasts.")
            with open(done_file_path, 'w') as f_done:
                f_done.write("done: no stage3 model")
            return True

        # --- Step 1: Load TIRP object ---
        with open(tirp_path, 'rb') as f_pkl:
            tirp_obj = pickle.load(f_pkl)

        # --- Step 2: Build the TEST durations table (event-free) ---
        start_dur = time.time()
        durations_df = build_forecast_durations(
            file_path=common_data['test_kl_path'],
            max_gap=max_gap,
            num_relations=num_relations,
            epsilon=epsilon,
            tirp_obj=tirp_obj.copy_tirp(),
            output_folder=current_tirp_dir,  # writes durations_merged_df.csv (test instances)
        )
        print(f"Test durations table: {durations_df.shape[0]} instance(s) in {time.time() - start_dur:.2f}s")

        # --- Step 3: Expand to per-(entity, t) matrix + metadata (no rows dropped) ---
        start_fm = time.time()
        X, meta, feature_names = build_forecast_inference_matrix(
            durations_df, common_data['target_stis'], horizon
        )
        print(f"Inference matrix: {X.shape[0]} rows, {X.shape[1]} features "
              f"in {time.time() - start_fm:.2f}s")

        if X.shape[0] == 0:
            print(f"TIRP {sanitized_id}: no test instances detected. Marking done, no forecasts.")
            with open(done_file_path, 'w') as f_done:
                f_done.write("done: no test rows")
            del tirp_obj
            return True

        # --- Step 4: Load the model and predict the symbol distribution per row ---
        with open(model_path, 'rb') as f_model:
            model = pickle.load(f_model)
        classes = np.asarray(model.classes_)

        start_pred = time.time()
        proba = model.predict_proba_matrix(X)  # (n_rows, n_classes), cols aligned to classes
        print(f"Predicted {proba.shape[0]} rows over {len(classes)} target symbols "
              f"in {time.time() - start_pred:.2f}s")

        # --- Step 5: Assemble + save the per-TIRP forecasts table ---
        proba_cols = {f"P_{int(c)}": proba[:, j] for j, c in enumerate(classes)}
        forecasts = pd.concat(
            [meta.reset_index(drop=True), pd.DataFrame(proba_cols)], axis=1
        )
        # The single most-likely forecast symbol per row (convenience for quick checks;
        # Stage 5 does the real cross-TIRP aggregation from the P_* columns).
        forecasts["pred_symbol"] = classes[proba.argmax(axis=1)]

        out_path = os.path.join(current_tirp_dir, "forecasts.csv.gz")
        forecasts.to_csv(out_path, index=False, compression="gzip")
        n_cov = int(forecasts["covered"].sum())
        print(f"Saved forecasts: {out_path} ({len(forecasts)} rows, {n_cov} with ground truth)")

        with open(done_file_path, 'w') as f_done:
            f_done.write("done")
        print(f"TIRP Done: {done_file_path}")

        del X, meta, durations_df, proba, forecasts, model, tirp_obj
        gc.collect()
        return True

    except Exception as e:
        sys.stderr.write("\n!!!!!!!!!! ERROR FORECASTING TIRP !!!!!!!!!!\n")
        sys.stderr.write(f"Parameters at failure: {debug_params}\n")
        sys.stderr.write(f"Exception: {e}\n")
        traceback.print_exc()
        sys.stderr.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print(f"CRITICAL ERROR in TIRP: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MARIO Stage 4: forecast the target symbol at t+HORIZON on the TEST set, "
                    "for a BATCH of TIRPs, using the Stage 3 models.")

    parser.add_argument("--abstraction_output_dir", required=True,
                        help="Stage 1 (Abstraction) output dir; expects Test/KL.txt.")
    parser.add_argument("--built_models_base_dir", required=True,
                        help="Stage 3 output dir containing tirp_<id>/models/*-CPML.pkl.")
    parser.add_argument("--prediction_output_dir", required=True,
                        help="Base output dir where tirp_<id>/forecasts.csv.gz are written.")
    parser.add_argument("--tirp_list_file", required=True,
                        help="Text file listing TIRP .pkl paths to process (one per line).")

    parser.add_argument("--max_gap", required=True, type=int, help="Max gap (from mining stage).")
    parser.add_argument("--num_relations", required=True, type=int, help="Number of relations (from mining stage).")
    parser.add_argument("--epsilon", required=True, type=int, help="Epsilon (from mining stage).")
    parser.add_argument("--target_variable", required=True, type=int,
                        help="TemporalPropertyID whose future abstracted state is forecast (must match Stage 3).")
    parser.add_argument("--horizon", required=True, type=int,
                        help="Forecast lead time (TimeStamp units): label at t + horizon (must match Stage 3).")

    args = parser.parse_args()

    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)

    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]

    if not tirp_paths:
        print("Warning: TIRP list file is empty.")
        sys.exit(0)

    print(f"Found {len(tirp_paths)} TIRPs to forecast in this batch.")

    try:
        common_data = prepare_batch_data(
            abstraction_output_dir=args.abstraction_output_dir,
            target_variable=args.target_variable,
        )

        for tirp_path in tirp_paths:
            process_single_tirp(
                tirp_path=tirp_path,
                built_models_base_dir=args.built_models_base_dir,
                prediction_output_dir=args.prediction_output_dir,
                common_data=common_data,
                max_gap=args.max_gap,
                num_relations=args.num_relations,
                epsilon=args.epsilon,
                horizon=args.horizon,
            )
            gc.collect()
            sys.stdout.flush()

    except Exception as e:
        sys.stderr.write(f"\nCRITICAL BATCH FAILURE: {e}\n")
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
