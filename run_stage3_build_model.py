import os
import argparse
import sys
import pickle
import pandas as pd
import time
from config import *
import traceback
import gc
import hashlib

# --- MARIO Stage 3 imports ---
# Event-free durations builder + the t+HORIZON forecast labeler live in the feature
# matrix module; the per-TIRP model is the (now multiclass) XGBoost CPML wrapper.
from CPM_Feature_Matrix.Create_feature_matrix import (
    build_forecast_durations,
    build_forecast_training_arrays,
)
from FCPM_Package.CPML import CPML


# --- Helper: parse a KL .txt into a long STI DataFrame ---------------------------

def txt_2_csv(data_path):
    """ Parses KarmaLego .txt data into a pandas DataFrame.

    Each interval line is `start,end,stateID,temporalPropertyID`. """
    rows = []
    entity_id = None
    with open(data_path, 'r') as file:
        lines = file.readlines()
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        if "," not in line and line.endswith(";"):
            entity_id = line.strip(";")
        elif entity_id is not None and line.endswith(";"):
            intervals = line.strip(";").split(";")
            for interval_str in intervals:
                if interval_str:
                    interval_data = interval_str.split(",")
                    if len(interval_data) == 4:
                        start_time, end_time, state_id, temporal_property_id = map(int, interval_data)
                        rows.append([entity_id, start_time, end_time, state_id, temporal_property_id])
    df = pd.DataFrame(rows, columns=["EntityID", "StartTime", "EndTime", "StateID", "TemporalPropertyID"])
    return df


def get_sanitized_tirp_id(tirp_object_file_path):
    """ Stable filesystem-safe id for a TIRP .pkl path (used for its output dir + done file). """
    try:
        tirp_filename = os.path.basename(tirp_object_file_path)
        tirp_id_base = os.path.splitext(tirp_filename)[0]
        sanitized_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in tirp_id_base)
        if not sanitized_id:
            hasher = hashlib.sha1(tirp_object_file_path.encode())
            sanitized_id = f"tirp_hash_{hasher.hexdigest()[:8]}"
        return sanitized_id
    except Exception as e:
        print(f"Warning: Could not generate TIRP ID: {e}")
        path_hash = hashlib.sha1(tirp_object_file_path.encode()).hexdigest()[:8]
        return f"tirp_error_id_{path_hash}"


# --- Batch-level common data -----------------------------------------------------

def prepare_batch_data(abstraction_output_dir, target_variable):
    """
    Common data shared by every TIRP in a batch. For MARIO this is minimal: the path
    to the training KL and the forecast target variable's STIs (from that same KL),
    used to label each (entity, t) row with the target symbol at t + HORIZON.

    Replaces the old FCPM prep (case/control window split, synthetic event insertion,
    combined KL files) entirely -- none of that exists in the forecasting pipeline.

    :param abstraction_output_dir: Stage 1 output dir (expects Train/KL.txt).
    :param target_variable: TemporalPropertyID whose future abstracted state is forecast.
    :return: dict with 'train_kl_path' and 'target_stis' (DataFrame
             [EntityID, StartTime, EndTime, StateID], filtered to the target variable).
    """
    train_kl_path = os.path.join(abstraction_output_dir, 'Train', 'KL.txt')
    if not os.path.exists(train_kl_path):
        raise FileNotFoundError(f"Train KL not found: {train_kl_path}")

    print(f"--- Preparing batch data: loading target STIs (TPID={target_variable}) from {train_kl_path} ---")
    all_stis = txt_2_csv(train_kl_path)
    target_stis = all_stis[all_stis["TemporalPropertyID"] == int(target_variable)][
        ["EntityID", "StartTime", "EndTime", "StateID"]
    ].copy()
    target_stis["EntityID"] = target_stis["EntityID"].astype(int)

    n_ent = target_stis["EntityID"].nunique()
    n_states = target_stis["StateID"].nunique()
    print(f"Target variable {target_variable}: {len(target_stis)} STIs over {n_ent} entities, "
          f"{n_states} distinct states.")
    if target_stis.empty:
        print(f"WARNING: no STIs for target variable {target_variable} in the training KL -- "
              f"every (entity, t) label will be undefined and all TIRPs will train on 0 rows.")

    return {"train_kl_path": train_kl_path, "target_stis": target_stis}


# --- Per-TIRP model build --------------------------------------------------------

def process_single_tirp(tirp_path, tirp_model_run_dir, common_data,
                        max_gap, num_relations, epsilon, horizon):
    """
    Build one per-TIRP MARIO forecast model:
      1. detect the TIRP's evolving prefixes on the training KL and extract the
         per-instance TIEP durations (build_forecast_durations, event-free);
      2. expand to a per-(entity, absolute t) matrix and label each row with the
         target symbol at t + horizon, dropping rows with no covering STI
         (build_forecast_training_arrays);
      3. train ONE multiclass model over that matrix and save it.

    Resumable via a stage3_build_<id>.done marker.
    """
    debug_params = {}
    try:
        sanitized_id = get_sanitized_tirp_id(tirp_path)
        current_tirp_dir = os.path.join(tirp_model_run_dir, f'tirp_{sanitized_id}')
        debug_params = {
            "tirp_path": tirp_path, "sanitized_id": sanitized_id,
            "current_tirp_dir": current_tirp_dir, "max_gap": max_gap,
            "num_relations": num_relations, "epsilon": epsilon, "horizon": horizon,
        }

        print(f"\n--- Processing TIRP: {sanitized_id} ---")
        done_file_path = os.path.join(current_tirp_dir, f'stage3_build_{sanitized_id}.done')
        if os.path.exists(done_file_path):
            print(f"Skipping TIRP {sanitized_id} (Already Done: {done_file_path})")
            return True
        os.makedirs(current_tirp_dir, exist_ok=True)

        # --- Step 1: Load TIRP object ---
        with open(tirp_path, 'rb') as f_pkl:
            tirp_obj = pickle.load(f_pkl)
        tirp_str = getattr(tirp_obj, 'to_string', lambda: 'UnknownTIRP')()

        # --- Step 2: Build the per-instance durations table (event-free) ---
        start_dur = time.time()
        durations_df = build_forecast_durations(
            file_path=common_data['train_kl_path'],
            max_gap=max_gap,
            num_relations=num_relations,
            epsilon=epsilon,
            tirp_obj=tirp_obj.copy_tirp(),
            output_folder=current_tirp_dir,
        )
        print(f"Durations table: {durations_df.shape[0]} instance(s) in {time.time() - start_dur:.2f}s")

        # --- Step 3: Expand + label at t + horizon ---
        start_fm = time.time()
        X, y, feature_names = build_forecast_training_arrays(
            durations_df, common_data['target_stis'], horizon
        )
        print(f"Feature matrix: {X.shape[0]} labeled rows, {X.shape[1]} features "
              f"in {time.time() - start_fm:.2f}s")

        if X.shape[0] == 0:
            print(f"TIRP {sanitized_id}: no labeled rows (no instance's t+horizon is covered "
                  f"by a target STI). Marking done without a model.")
            with open(done_file_path, 'w') as f_done:
                f_done.write("done: no labeled rows")
            del tirp_obj
            return True

        # --- Step 4: Train + save the multiclass forecast model ---
        start_build = time.time()
        model = CPML()
        model.fit_matrix(X, y, feature_names)
        n_classes = 0 if model.classes_ is None else len(model.classes_)
        print(f"Trained forecast model ({n_classes} target symbols) in {time.time() - start_build:.2f}s")

        model_dir = os.path.join(current_tirp_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{tirp_str}-CPML.pkl")
        with open(model_path, 'wb') as f_out:
            pickle.dump(model, f_out)
        print(f"Saved model: {model_path}")

        with open(done_file_path, 'w') as f_done:
            f_done.write("done")
        print(f"TIRP Done: {done_file_path}")

        del X, y, durations_df, model, tirp_obj
        gc.collect()
        return True

    except Exception as e:
        sys.stderr.write("\n!!!!!!!!!! ERROR PROCESSING TIRP !!!!!!!!!!\n")
        sys.stderr.write(f"Parameters at failure: {debug_params}\n")
        sys.stderr.write(f"Exception: {e}\n")
        traceback.print_exc()
        sys.stderr.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print(f"CRITICAL ERROR in TIRP: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARIO Stage 3: build a forecast model for a BATCH of TIRPs.")

    parser.add_argument("--abstraction_output_dir", required=True,
                        help="Stage 1 (Abstraction) output dir; expects Train/KL.txt.")
    parser.add_argument("--tirp_model_run_dir", required=True,
                        help="Base output dir where tirp_<id>/ folders are created.")
    parser.add_argument("--tirp_list_file", required=True,
                        help="Text file listing TIRP .pkl paths to process (one per line).")

    parser.add_argument("--max_gap", required=True, type=int, help="Max gap (from mining stage).")
    parser.add_argument("--num_relations", required=True, type=int, help="Number of relations (from mining stage).")
    parser.add_argument("--epsilon", required=True, type=int, help="Epsilon (from mining stage).")
    parser.add_argument("--target_variable", required=True, type=int,
                        help="TemporalPropertyID whose future abstracted state is forecast.")
    parser.add_argument("--horizon", required=True, type=int,
                        help="Forecast lead time (TimeStamp units): label at t + horizon.")

    args = parser.parse_args()

    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)

    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]

    if not tirp_paths:
        print("Warning: TIRP list file is empty.")
        sys.exit(0)

    print(f"Found {len(tirp_paths)} TIRPs to process in this batch.")

    try:
        common_data = prepare_batch_data(
            abstraction_output_dir=args.abstraction_output_dir,
            target_variable=args.target_variable,
        )

        for i, tirp_path in enumerate(tirp_paths):
            process_single_tirp(
                tirp_path=tirp_path,
                tirp_model_run_dir=args.tirp_model_run_dir,
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
