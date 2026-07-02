import os
import argparse
import sys
import pickle
import pandas as pd
import time
from config import *
import shutil
import tempfile
import traceback
import gc
import hashlib

# --- Assume necessary imports succeed ---
# Adjust paths based on your project structure
from CPM_Feature_Matrix.Run_create_feature_matrix import create_feature_matrix_for_CPM
from FCPM_Package.build_and_evaluate import build_and_evaluate
from FCPM_Package.CPML import CPML # Ensure CPML.py is in the same directory or accessible




# --- Helper Functions (Simplified - Copied from Stage 2) ---

def txt_2_csv(data_path):
    """ Parses KarmaLego .txt data into a pandas DataFrame. (Simplified) """
    rows = []
    entity_id = None
    with open(data_path, 'r') as file:
        lines = file.readlines()
    for line in lines[2:]:
        line = line.strip()
        if not line: continue
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

def csv_2_txt(df, output_path, number_of_entities):
    """ Converts a DataFrame back to the KarmaLego .txt format. (Simplified) """
    with open(output_path, 'w') as file:
        file.write("startToncepts\n")
        file.write(f"numberOfEntities,{int(number_of_entities)}\n")
        grouped = df.groupby("EntityID")
        for entity_id, group in grouped:
            file.write(f"{entity_id};\n")
            intervals = []
            for _, row in group.iterrows():
                interval = (
                    f"{int(row['StartTime'])},{int(row['EndTime'])},"
                    f"{int(row['StateID'])},{int(row['TemporalPropertyID'])}"
                )
                intervals.append(interval)
            if intervals:
                file.write(";".join(intervals) + ";\n")
            else:
                file.write("\n")



def split_data(data_cls0_path, data_cls01_path, target_dir, window_size, event_symbol):
    """
    Splits data. Case intervals overlapping the window are processed:
    - A segment (up to 1.2 * window_size) at the end of the window becomes 'valid'.
    - Other parts of the original interval become 'early'.
    Adds synthetic events. Samples controls.
    (Simplified - assumes valid inputs).
    """
    os.makedirs(target_dir, exist_ok=True)
    event_symbol_int = int(event_symbol)

    df_controls_orig = txt_2_csv(data_cls0_path)
    df_cases_orig = txt_2_csv(data_cls01_path)

    # --- Process Cases: Split into early and valid intervals (without adding event yet) ---
    valid_intervals_list = [] # Intervals within the window
    early_cases_list = []     # Intervals before the window
    case_entities_for_event = {} # Store max_end_time for valid interval entities

    if not df_cases_orig.empty:
        # Calculate max_end_time_for_entity once per entity using transform
        df_cases_orig['max_end_time_for_entity'] = df_cases_orig.groupby('EntityID')['EndTime'].transform('max')
        df_cases_orig['cutoff_time'] = df_cases_orig['max_end_time_for_entity'] - window_size

        for entity_id, group in df_cases_orig.groupby("EntityID"):
            if group.empty: continue
            
            # Use the pre-calculated cutoff_time for this entity
            # All rows in 'group' will have the same 'max_end_time_for_entity' and 'cutoff_time'
            entity_cutoff_time = group['cutoff_time'].iloc[0]
            entity_max_end_time = group['max_end_time_for_entity'].iloc[0]

            for _, row in group.iterrows():
                s_orig = int(row["StartTime"])
                e_orig = int(row["EndTime"])
                state_id = row["StateID"]
                tp_id = row["TemporalPropertyID"]

                # Case 1: Interval ends completely before the window starts
                if e_orig <= entity_cutoff_time:
                    early_cases_list.append(pd.DataFrame([{"EntityID": entity_id, "StartTime": s_orig, "EndTime": e_orig, "StateID": state_id, "TemporalPropertyID": tp_id}]))
                # Case 2: Interval straddles the cutoff_time (s_orig < entity_cutoff_time AND e_orig >= entity_cutoff_time)
                else:                    
                    # Part before cutoff_time always goes to early
                    if s_orig < entity_cutoff_time:
                        early_cases_list.append(pd.DataFrame([{"EntityID": entity_id, "StartTime": s_orig, "EndTime": entity_cutoff_time - 1, "StateID": state_id, "TemporalPropertyID": tp_id}]))
                        valid_intervals_list.append(pd.DataFrame([{"EntityID": entity_id, "StartTime": entity_cutoff_time, "EndTime": e_orig, "StateID": state_id, "TemporalPropertyID": tp_id}]))
                        case_entities_for_event[entity_id] = entity_max_end_time
                    else:
                        # Valid part is not too long, take it from cutoff_time to e_orig
                        valid_intervals_list.append(pd.DataFrame([{"EntityID": entity_id, "StartTime": s_orig, "EndTime": e_orig, "StateID": state_id, "TemporalPropertyID": tp_id}]))
                        case_entities_for_event[entity_id] = entity_max_end_time

    df_early_cases = pd.concat(early_cases_list, ignore_index=True) if early_cases_list else pd.DataFrame(columns=df_cases_orig.columns)
    df_valid_intervals = pd.concat(valid_intervals_list, ignore_index=True) if valid_intervals_list else pd.DataFrame(columns=df_cases_orig.columns)

    # --- Create intermediate new controls (original controls + early cases) ---
    df_new_controls_intermediate = pd.concat([df_controls_orig, df_early_cases], ignore_index=True)

    # --- Add synthetic event to ALL entities in the intermediate new controls group ---
    new_controls_events_list = []
    for entity_id, group in df_new_controls_intermediate.groupby("EntityID"):
         # Assume 'EndTime' is numeric and present
        max_end_time_in_group = group["EndTime"].max()
        new_controls_events_list.append(pd.DataFrame({
            "EntityID": [entity_id], "StartTime": [int(max_end_time_in_group + 1)],
            "EndTime": [int(max_end_time_in_group + 2)], "StateID": [event_symbol_int],
            "TemporalPropertyID": [event_symbol_int]
        }))
    df_new_controls_events = pd.concat(new_controls_events_list, ignore_index=True) if new_controls_events_list else pd.DataFrame()

    # Combine intermediate controls with their events
    df_new_controls_final = pd.concat([df_new_controls_intermediate, df_new_controls_events], ignore_index=True)

    # --- Add synthetic event to the advanced cases group ---
    advanced_cases_events_list = []
    # Iterate using the stored max times for entities that had valid intervals
    for entity_id, max_end_time in case_entities_for_event.items():
         advanced_cases_events_list.append(pd.DataFrame({
             "EntityID": [entity_id], "StartTime": [int(max_end_time + 1)],
             "EndTime": [int(max_end_time + 2)], "StateID": [event_symbol_int],
             "TemporalPropertyID": [event_symbol_int]
         }))
    df_advanced_cases_events = pd.concat(advanced_cases_events_list, ignore_index=True) if advanced_cases_events_list else pd.DataFrame()

    # Combine valid intervals with their events
    df_cases_advanced_final = pd.concat([df_valid_intervals, df_advanced_cases_events], ignore_index=True)

    # --- Calculate entity counts and save final DataFrames ---
    num_entities_new_controls = df_new_controls_final["EntityID"].nunique() if not df_new_controls_final.empty else 0
    num_entities_advanced_cases = df_cases_advanced_final["EntityID"].nunique() if not df_cases_advanced_final.empty else 0

    output_new_controls_path = os.path.join(target_dir, 'KL-class-0.0.txt')
    output_advanced_cases_path = os.path.join(target_dir, 'KL-class-1.0.txt')

    csv_2_txt(df_new_controls_final, output_new_controls_path, num_entities_new_controls)
    csv_2_txt(df_cases_advanced_final, output_advanced_cases_path, num_entities_advanced_cases)

    return output_advanced_cases_path, output_new_controls_path


def add_event_for_test(test_cls0_path, test_cls1_path, target_dir, event_symbol):
    """
    Adds a synthetic event symbol after the maximum end time for all entities
    (both class 0 and class 1) in the test set and combines them into a
    single KL file. Assumes valid inputs and numeric EndTime.

    Args:
        test_cls0_path (str): Path to the class 0 test data KL file.
        test_cls1_path (str): Path to the class 1 test data KL file.
        target_dir (str): Directory where the combined output KL file will be saved.
        event_symbol (str or int): The event symbol to add.

    Returns:
        str: Path to the combined output KL file.
    """
    event_symbol_int = int(event_symbol)

    # Read input test KL files
    df_controls_test = txt_2_csv(test_cls0_path)
    df_cases_test = txt_2_csv(test_cls1_path)

    modified_controls_list = []
    modified_cases_list = []

    # Process class 0 entities: Append original data + synthetic event
    for entity_id, group in df_controls_test.groupby("EntityID"):
        modified_controls_list.append(group) # Append original intervals
        max_end_time = group["EndTime"].max()
        modified_controls_list.append(pd.DataFrame({ # Append synthetic event
            "EntityID": [entity_id], "StartTime": [int(max_end_time + 1)],
            "EndTime": [int(max_end_time + 2)], "StateID": [event_symbol_int],
            "TemporalPropertyID": [event_symbol_int]
        }))

    # Process class 1 entities: Append original data + synthetic event
    for entity_id, group in df_cases_test.groupby("EntityID"):
        modified_cases_list.append(group) # Append original intervals
        max_end_time = group["EndTime"].max()
        modified_cases_list.append(pd.DataFrame({ # Append event
            "EntityID": [entity_id], "StartTime": [int(max_end_time + 1)],
            "EndTime": [int(max_end_time + 2)], "StateID": [event_symbol_int],
            "TemporalPropertyID": [event_symbol_int]
        }))
        modified_cases_list.append(pd.DataFrame({ # Append event
            "EntityID": [entity_id], "StartTime": [int(max_end_time + 1)],
            "EndTime": [int(max_end_time + 2)], "StateID": [event_symbol_int+1],
            "TemporalPropertyID": [event_symbol_int+1]
        }))

    # Concatenate modified controls and cases separately
    df_controls_modified = pd.concat(modified_controls_list, ignore_index=True) if modified_controls_list else pd.DataFrame(columns=df_controls_test.columns)
    df_cases_modified = pd.concat(modified_cases_list, ignore_index=True) if modified_cases_list else pd.DataFrame(columns=df_cases_test.columns)

    # Combine modified controls and modified cases
    df_combined_test = pd.concat([df_controls_modified, df_cases_modified], ignore_index=True)

    # Calculate total unique entities for the header
    total_entities_combined = df_combined_test["EntityID"].nunique()

    # Define the output path
    output_combined_path = os.path.join(target_dir, 'KL_test.txt') # Changed filename slightly for clarity

    # Save the combined DataFrame to a single KL txt file
    csv_2_txt(df_combined_test, output_combined_path, total_entities_combined)

    return output_combined_path



def get_total_entities(data_path):
    """ Reads number of entities from KL file header. """
    with open(data_path, 'r') as file:
        lines = file.readlines()
    # Assumes header is present and correct
    number_of_entities = lines[1].split(",")[-1]
    return int(number_of_entities)


# --- Copied Helper Function: Sanitization ---
def get_sanitized_tirp_id(tirp_object_file_path):
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



def prepare_batch_data(abstraction_output_dir, tirp_model_run_dir, max_gap, event_symbol, window_size_ratio):
    """
    Performs the common data preparation steps for a batch of TIRPs.
    Returns the paths to the prepared data files.
    """
    print("--- Preparing Common Data for Batch ---")
    
    # Create a temporary directory for the batch data
    batch_temp_dir = os.path.join(tirp_model_run_dir, f'batch_data_{os.getpid()}_{int(time.time())}')
    os.makedirs(batch_temp_dir, exist_ok=True)
    
    # Create a sub-directory for the train combined file to prevent overriding KL_test.txt
    train_combined_dir = os.path.join(batch_temp_dir, 'train_combined')
    os.makedirs(train_combined_dir, exist_ok=True)
    
    print(f"Batch temporary directory: {batch_temp_dir}")
    
    try:
        # --- Step 2: Define Paths & Calculate Window Size ---
        original_cls0_path = os.path.join(abstraction_output_dir, 'Train', 'KL-class-0.0.txt')
        original_cls1_path = os.path.join(abstraction_output_dir, 'Train', 'KL-class-1.0.txt')
        test_cls0_path = os.path.join(abstraction_output_dir, 'Test', 'KL-class-0.0.txt')
        test_cls1_path = os.path.join(abstraction_output_dir, 'Test', 'KL-class-1.0.txt')
        
        # The entity-class-relations.csv already exists in the Train folder
        train_class_file_path = os.path.join(abstraction_output_dir, 'Train', 'entity-class-relations.csv')
        
        # Create combined test file (Exact original behavior)
        test_file_path = add_event_for_test(test_cls0_path, test_cls1_path, batch_temp_dir, event_symbol)

        # Create combined train file using the exact same function, directed to the sub-directory
        train_combined_file_path = add_event_for_test(original_cls0_path, original_cls1_path, train_combined_dir, event_symbol)
        
        # Window size calculation
        window_size = int(window_size_ratio * max_gap)
        print(f"Calculated window size for splitting: {window_size} (ratio={window_size_ratio}, max_gap={max_gap})")

        # --- Step 3: Split Data ---
        print("Splitting original training data for feature matrix...")
        split_cls1_path, split_cls0_path = split_data(
            original_cls0_path, original_cls1_path, batch_temp_dir,
            window_size=window_size, event_symbol=event_symbol
        )
        print(f"Split data saved in: {batch_temp_dir}")
        
        # Save as CSVs
        cls1_df = txt_2_csv(split_cls1_path)
        cases_csv_path = os.path.join(batch_temp_dir, 'cases.csv')
        cls1_df.to_csv(cases_csv_path, index=False)
        
        cls0_df = txt_2_csv(split_cls0_path)
        controls_csv_path = os.path.join(batch_temp_dir, 'controls.csv')
        cls0_df.to_csv(controls_csv_path, index=False)
        
        # --- Step 5: Get Total Entities ---
        print("Getting total entity count...")
        num_entities_after_split_cases = get_total_entities(split_cls1_path)
        num_entities_after_split_controls = get_total_entities(split_cls0_path)
        total_entities = num_entities_after_split_cases + num_entities_after_split_controls
        print(f"Total entities: {total_entities}")
        
        return {
            "batch_temp_dir": batch_temp_dir,
            "split_cls0_path": split_cls0_path,
            "split_cls1_path": split_cls1_path,
            "test_file_path": test_file_path,
            "cases_csv_path": cases_csv_path,
            "controls_csv_path": controls_csv_path,
            "train_combined_file_path": train_combined_file_path,  # Passed to validation
            "train_class_file_path": train_class_file_path,        # Passed to validation
            "window_size": window_size                             # Passed to validation
        }
        
    except Exception as e:
        print(f"ERROR in prepare_batch_data: {e}")
        # If preparation fails, we can't process anything
        shutil.rmtree(batch_temp_dir, ignore_errors=True)
        raise e

def process_single_tirp(tirp_path, tirp_model_run_dir, common_data, 
                       max_gap, num_relations, epsilon, event_symbol,build_cpml=False):
    """
    Processes a single TIRP using the pre-prepared common data.
    """
    debug_params = {} # Initialize empty
    try:
        sanitized_id = get_sanitized_tirp_id(tirp_path)
        current_tirp_dir = os.path.join(tirp_model_run_dir, f'tirp_{sanitized_id}')
        
        # Populate debug params
        debug_params = {
            "tirp_path": tirp_path,
            "sanitized_id": sanitized_id,
            "current_tirp_dir": current_tirp_dir,
            "max_gap": max_gap,
            "num_relations": num_relations,
            "epsilon": epsilon,
            "event_symbol": event_symbol,
            "common_data_paths": common_data
        }
        
        print(f"\n--- Processing TIRP: {sanitized_id} ---")
        os.makedirs(current_tirp_dir, exist_ok=True)
        target_dir = current_tirp_dir # Use as base for outputs

        # --- Step 0: Check if already done ---
        done_file_path = os.path.join(target_dir, f'stage3_build_{sanitized_id}.done')
        if os.path.exists(done_file_path):
            print(f"Skipping TIRP {sanitized_id} (Already Done: {done_file_path})")
            return True

        # --- Step 1: Load TIRP Object ---
        with open(tirp_path, 'rb') as f_pkl:
            tirp_obj = pickle.load(f_pkl)
        tirp_str = getattr(tirp_obj, 'to_string', lambda: 'UnknownTIRP')()
        
        # --- Step 4: Create Feature Matrix ---
        start_fm_time = time.time()
        create_feature_matrix_for_CPM(
            class0_file_path=common_data['split_cls0_path'],
            class1_file_path=common_data['split_cls1_path'],
            test_file_path=common_data['test_file_path'],
            max_gap=max_gap,
            num_relations=num_relations,
            epsilon=epsilon,
            tirp_obj=tirp_obj,
            event_symbol=event_symbol,
            output_folder_base=target_dir
        )
        print(f"Feature Matrix created in {time.time() - start_fm_time:.2f}s")
        
        # --- Step 6: Define TIRP Config ---
        tirp_config = {
            "tirp_name": tirp_str,
            "fcpm_cls0_path": os.path.join(target_dir, 'class0'),
            "fcpm_cls1_path": os.path.join(target_dir, 'class1'),
            "tte_train_path": os.path.join(target_dir, 'class1', 'TTE_feature_matrix.csv'),
            "test_path": os.path.join(target_dir, 'test', 'final_test_feature_matrix.csv')
        }
        
        # --- Step 7: Build and Evaluate FCPM ---
        start_build_time = time.time()
        tirp_model = build_and_evaluate(
            tirp_config=tirp_config,
            event_symbol=event_symbol,
            base_dir=target_dir,
            cases_data_path=common_data['cases_csv_path'],
            controls_data_path=common_data['controls_csv_path']
        )
        
        if tirp_model and hasattr(tirp_model, 'build_model_and_predict'):
            tirp_model.build_model_and_predict()
            
            # --- NEW: CPML Matrix Building & Training ---
            if build_cpml:
                print(f"Building CPML model for TIRP {sanitized_id}...")
                
                # 1. Prepare Shared Inference Tables (extracted from validation logic)
                from CPM_Feature_Matrix.Create_feature_matrix import run_test_table, build_cpml_training_arrays

                train_inference_dir = os.path.join(target_dir, "train_inference")
                os.makedirs(train_inference_dir, exist_ok=True)
                
                run_test_table(
                    file_path=common_data['train_combined_file_path'],
                    max_gap=max_gap,
                    num_relations=num_relations,
                    epsilon=epsilon,
                    tirp_obj=tirp_obj.copy_tirp(),
                    event_symbol=event_symbol,
                    output_folder=train_inference_dir
                )
                
                durations_path = os.path.join(train_inference_dir, "durations_merged_df.csv")

                if os.path.exists(durations_path):
                    # 2. Load the per-instance duration table (small: one row per instance).
                    durations_merged_df = pd.read_csv(durations_path, low_memory=False)

                    # 3. Map each entity to its true outcome class for the labels.
                    class_df = pd.read_csv(common_data['train_class_file_path'], dtype={"EntityID": int, "ClassID": int})
                    class_map = dict(zip(class_df["EntityID"], class_df["ClassID"]))

                    # 4. Build the per-timestamp feature matrix as a compact integer NumPy
                    #    array (no list-of-dicts / DataFrame) so RAM stays low on big TIRPs.
                    print("Building continuous CPML feature matrix as a compact integer array...")
                    X_train, y_train, feature_names = build_cpml_training_arrays(durations_merged_df, class_map)

                    if X_train.shape[0] > 0:
                        # 5. Train CPML Model
                        cpml_model = CPML()
                        cpml_model.fit_matrix(X_train, y_train, feature_names)

                        # 6. Save CPML Model
                        cpml_model_path = os.path.join(target_dir, 'models', f"{tirp_str}-CPML.pkl")
                        os.makedirs(os.path.dirname(cpml_model_path), exist_ok=True)
                        with open(cpml_model_path, 'wb') as f_out:
                            pickle.dump(cpml_model, f_out)

                        print(f"CPML model trained and saved successfully.")

                    # 7. CRITICAL: Memory Cleanup
                    del durations_merged_df
                    del X_train
                    del y_train
                    gc.collect()
                    
            # --------------------------------------------
            
            # Create Done file for Stage 3
            done_file_path = os.path.join(target_dir, f'stage3_build_{sanitized_id}.done')
            with open(done_file_path, 'w') as f_done:
                f_done.write("done")
            print(f"TIRP Done: {done_file_path}")
            
            del tirp_model
            del tirp_obj
            return True
        else:
             raise RuntimeError("build_and_evaluate returned invalid model object")
    except Exception as e:
        sys.stderr.write("\n!!!!!!!!!! ERROR PROCESSING TIRP !!!!!!!!!!\n")
        sys.stderr.write(f"Parameters at failure: {debug_params}\n")
        sys.stderr.write("-" * 40 + "\n")
        sys.stderr.write(f"Exception: {e}\n")
        traceback.print_exc() # Writes to stderr
        sys.stderr.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # Also print to stdout for convenience in .out logs
        print(f"CRITICAL ERROR in TIRP: {e}")
        raise e 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage 3: Build and evaluate model for a BATCH of TIRPs.")

    parser.add_argument("--abstraction_output_dir", required=True, help="Path to the output directory of the corresponding Stage 1 (Abstraction) run.")
    parser.add_argument("--tirp_model_run_dir", required=True, help="Base output directory for the feature_matrix folder.")
    parser.add_argument("--tirp_list_file", required=True, help="Path to text file containing list of TIRP .pkl paths to process.")
    
    parser.add_argument("--max_gap", required=True, type=int, help="Maximum gap parameter (from mining stage).")
    parser.add_argument("--num_relations", required=True, type=int, help="Number of relations (from mining stage).")
    parser.add_argument("--epsilon", required=True, type=int, help="Epsilon parameter (from mining stage).")
    parser.add_argument("--event_symbol", required=True, help="Event symbol.")
    parser.add_argument("--window_size_ratio", type=float, default=WINDOW_SIZE_RATIO, help="Ratio to calculate window size for data split in Stage 3.")
    parser.add_argument("--build_cpml", action="store_true", help="Flag to build the CPML model and its bulk feature matrix.")

    args = parser.parse_args()

    # Read the list of TIRPs
    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)
        
    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]
        
    if not tirp_paths:
        print("Warning: TIRP list file is empty.")
        sys.exit(0)
        
    print(f"Found {len(tirp_paths)} TIRPs to process in this batch.")
    
    # 1. Prepare Common Data
    common_data = None
    try:
        common_data = prepare_batch_data(
            abstraction_output_dir=args.abstraction_output_dir, 
            tirp_model_run_dir=args.tirp_model_run_dir, 
            max_gap=args.max_gap, 
            event_symbol=args.event_symbol, 
            window_size_ratio=args.window_size_ratio
        )
        
        # 2. Process TIRPs Loop
        for i, tirp_path in enumerate(tirp_paths):
            process_single_tirp(
                tirp_path=tirp_path,
                tirp_model_run_dir=args.tirp_model_run_dir,
                common_data=common_data,
                max_gap=args.max_gap,
                num_relations=args.num_relations,
                epsilon=args.epsilon,
                event_symbol=args.event_symbol,
                build_cpml=args.build_cpml
            )
            
            # Clean up memory
            gc.collect()
            print(f"RAM Cleaned (gc.collect)")
            sys.stdout.flush() # Ensure progress is written
            
    except Exception as e:
        sys.stderr.write(f"\nCRITICAL BATCH FAILURE: {e}\n")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup batch temp dir
        if common_data and 'batch_temp_dir' in common_data and os.path.exists(common_data['batch_temp_dir']):
             print(f"Cleaning up batch temp dir: {common_data['batch_temp_dir']}")
             shutil.rmtree(common_data['batch_temp_dir'], ignore_errors=True)

    sys.exit(0)