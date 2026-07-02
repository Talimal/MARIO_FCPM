import pandas as pd
import os
import argparse
import sys
import time
import config 
from Tirp_selection import TirpSelection
import random
from experiment_params import Tirp_selection_methods
from config import *

# --- Assume necessary imports succeed ---
from New_KarmaLego_Framework import RunKarmaLego
# Assume TIRP object has a functioning save_tirp_object method


from pathlib import Path
from typing import Dict, Tuple, Union

def parse_experiment_path(path_str: str) -> Tuple[str,
                                                  Dict[str, Union[int, float, str, bool]],
                                                  str]:
    """
    Parse an experiment results path and build the CSV filename that concatenates
    all parameters.  Returns:
        1) base_path  – the path up to (and including) the fold directory
        2) params     – dictionary of all parameters
        3) csv_path   – full path (base_path / <combined-name>.csv)
    """
    p = Path(path_str).expanduser().resolve()
    parts = p.parts

    # locate "fold_*"
    try:
        fold_idx = next(i for i, part in enumerate(parts) if part.startswith("fold_"))
    except StopIteration:
        raise ValueError("No directory that starts with 'fold_' found")

    # need at least two sub-directories with parameters after the fold dir
    if fold_idx + 2 >= len(parts):
        raise ValueError("Path must contain two parameter sub-directories after the fold directory")

    base_path = Path(*parts[:fold_idx])   # Path object (keep for joining later)
    fold_num  = int(parts[fold_idx].split("_", 1)[1])

    # ----- discretization params -----
    disc_dir  = parts[fold_idx + 1]                # abs_b=3-equal-frequency-ig=2
    disc_tokens = disc_dir.split("-")

    disc_params: Dict[str, str] = {}
    for tok in disc_tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            disc_params[k] = v
        else:
            disc_params["discretization_type"] = tok

    # ----- mining params -----
    mine_dir  = parts[fold_idx + 2]                # mine_e=0-mg=28-mvs=0_dot_5-rel=7-sf=False
    mine_params = dict(tok.split("=", 1) for tok in mine_dir.split("-"))

    # build final dict
    params: Dict[str, Union[int, float, str, bool]] = {
        "fold":                fold_num,
        "abs_b":               int(disc_params["abs_b"]),
        "discretization_type": disc_params["discretization_type"],
        "ig":                  int(disc_params["ig"]),
        "mine_e":              float(mine_params["mine_e"]),
        "mg":                  int(mine_params["mg"]),
        "mvs":                 float(mine_params["mvs"].replace("_dot_", ".")),
        "rel":                 int(mine_params["rel"]),
        "sf":                  mine_params["sf"].lower() == "true",
    }

    # ---------- compose CSV filename ----------
    # order the fields so filenames are deterministic/readable
    name_parts = [
        f"fold_{params['fold']}",
        f"abs_b={params['abs_b']}",
        params['discretization_type'],
        f"ig={params['ig']}",
        f"mine_e={params['mine_e']}",
        f"mg={params['mg']}",
        f"mvs={params['mvs']}",
        f"rel={params['rel']}",
        f"sf={params['sf']}",
    ]
    csv_filename = "-".join(str(part) for part in name_parts) + ".csv"
    csv_path     = (base_path /'tirp_selection'/ csv_filename).as_posix()
    tirp_selection_dir = base_path / 'tirp_selection'
    tirp_selection_dir.mkdir(parents=True, exist_ok=True)   

    return base_path.as_posix(), params, csv_path



# --- Helper Functions (Simplified) ---

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
                        # Filter out invalid state IDs (-1) early in the pipeline
                        if state_id != -1:
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
    # split data function from stage 3 with interval splitting 
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

    return output_advanced_cases_path



def Tirp_selection(tirps, event_symbol):
    """ Selects TIRPs that end with the specified event symbol and sorts them by vertical support. """
    selected_tirps = []
    event_symbol_str = str(event_symbol)
    # Initial check for None is minimal and reasonable
    if tirps is None:
        return []

    # Filter TIRPs based on criteria
    for tirp in tirps:
        # Assume necessary attributes exist for simplicity
        if hasattr(tirp, '_symbols') and tirp._symbols and hasattr(tirp, 'size'):
            last_symbol = str(tirp._symbols[-1])
            # Skip TIRPs with size 2 or less, or if last symbol doesn't match
            if tirp.size > 2 and last_symbol == event_symbol_str:
                selected_tirps.append(tirp)

    # Sort the filtered list by vertical support (descending)
    # Assumes get_vertical_support() method exists and returns a sortable numeric value
    selected_tirps.sort(key=lambda t: t.get_vertical_support(), reverse=True)

    return selected_tirps


# --- Main Mining Function (Simplified) ---

def run_single_mining_task(abstraction_output_dir, mining_run_dir, tirp_objects_output_dir,
                           mvs, max_gap, relations, skip_followers, epsilon, event_symbol):
    """ Runs the simplified TIRP mining process. """
    print(f"--- Starting Stage 2: TIRP Mining (Simplified) ---")
    print(f"Params: mvs={mvs}, max_gap={max_gap}, relations={relations}, skip_followers={skip_followers}, epsilon={epsilon}, event={event_symbol}")
    print(f"Input: {abstraction_output_dir}")
    print(f"Output: {tirp_objects_output_dir}")
    start_total_time = time.time()

    # --- Define Paths ---
    input_cls0_path = os.path.join(abstraction_output_dir,'Train', 'KL-class-0.0.txt')
    input_cls1_path = os.path.join(abstraction_output_dir, 'Train','KL-class-1.0.txt')
    split_kl_dir = os.path.join(mining_run_dir, 'KL_data')
    os.makedirs(tirp_objects_output_dir, exist_ok=True) # Ensure final output dir exists

    # --- Step 1: Split Data ---
    print("Step 1: Splitting data...")
    karmalego_input_dir = split_data(input_cls0_path, input_cls1_path, split_kl_dir, window_size=max_gap*WINDOW_SIZE_RATIO, event_symbol=event_symbol)
    print(f"Data splitting complete. Split files in: {karmalego_input_dir}")

    # --- Step 2: Run KarmaLego ---
    print("Step 2: Running KarmaLego algorithm...")
    start_kl_time = time.time()
    lego_result, karma_result = RunKarmaLego.runKarmaLego(
                                      time_intervals_path=karmalego_input_dir,
                                      min_ver_support=mvs,
                                      num_relations=relations,
                                      max_gap=max_gap,
                                      epsilon=epsilon,
                                      skip_followers=skip_followers,
                                      max_tirp_length=8,
                                      # Defaults for other params assumed sufficient
                                      label='KarmaLegoRun',
                                      output_path=mining_run_dir, # For potential intermediate outputs
                                      print_instances=False, # Reduce verbosity
                                      print_params=False,    # Reduce verbosity
                                      processes_num=4, # Use 4 processes for parallelism
                                      skip_same_variable=config.SKIP_SAME_VARIABLE, # Skip same variable indexing
                                      )
    end_kl_time = time.time()
    print(f"KarmaLego finished. Time: {end_kl_time - start_kl_time:.2f} seconds.")

    frequent_tirps = []
    if lego_result and hasattr(lego_result, 'frequent_tirps'):
        frequent_tirps = lego_result.frequent_tirps
    print(f"KarmaLego found {len(frequent_tirps)} frequent TIRPs.")

        # --- Step 3: Score TIRPs and Select/Save based on Selection Flags ---
    print("Step 3: Scoring TIRPs and selecting/saving based on selection flags...")

    # Define scoring methods to calculate (must match methods used in TirpSelection)
    stage2_valid_methods = ['diff_horizontal_support','diff_mean_duration', 'diff_vertical_support','mean_squared_mmd','mean_squared_vs','all','random']
    methods_to_calc = [m for m in Tirp_selection_methods if m in stage2_valid_methods]

    # Instantiate the selector (passing the limit for flag calculation)
    selector = TirpSelection(
        event_symbol=event_symbol,
        scoring_methods=methods_to_calc,
    )

    # Prepare potential extra arguments for scoring
    kwargs_for_scoring = {
        'class0_data_path': os.path.join(split_kl_dir, 'KL-class-0.0.txt'),
        'class1_data_path': os.path.join(split_kl_dir, 'KL-class-1.0.txt'),
        'detection_params': {'max_gap': max_gap, 'epsilon': epsilon, 'relations': relations}
    }

    # Calculate scores and get DataFrame with selection flags
    scores_df = selector.calculate_scores(
        tirps_list=frequent_tirps,
        **kwargs_for_scoring
    )

    # (Optional) Save the comprehensive scoring results to CSV
    if not scores_df.empty:
        # Create path relative to the specific mining run directory
        scores_csv_path = os.path.join(mining_run_dir, "tirp_selection_scores.csv")
        try:
            # # Save without the 'TIRP_Object' column
            # _ , _ , tirp_selection_phat = parse_experiment_path(mining_run_dir)
            # scores_df.drop(columns=['TIRP_Object']).to_csv(tirp_selection_phat, index=False)

            scores_df.drop(columns=['TIRP_Object']).to_csv(scores_csv_path, index=False)
            # print(f"Saved detailed scores for {len(scores_df)} filtered TIRPs to {scores_csv_path}") # Removed print
        except Exception as e:
            print(f"Warning: Could not save scores CSV {scores_csv_path}: {e}")

    # --- New Logic: Select TIRP Objects based on ANY 'Selected_*' flag == 1 ---
    tirps_to_save = [] # Initialize empty list

    if not scores_df.empty: # Selection only relevant if limit > 0
        # Find columns starting with 'Selected_'
        selected_columns = [col for col in scores_df.columns if col.startswith('Binary_')]

        if selected_columns: # Check if any selection columns exist
            # Create a boolean Series: True if any 'Selected_*' column is 1 for that row
            filter_condition = scores_df[selected_columns].any(axis=1)

            # Apply the filter to get rows where at least one flag is 1
            selected_rows_df = scores_df[filter_condition]

            # Extract the TIRP objects from the filtered DataFrame
            tirps_to_save = selected_rows_df['TIRP_Object'].tolist()
        # else: No 'Selected_*' columns found, tirps_to_save remains empty



    print(f"Selected {len(tirps_to_save)} unique TIRPs based on being in top for at least one method.")

    # Save only the selected unique TIRP objects for Stage 3
    saved_count = 0
    if tirps_to_save:
        os.makedirs(tirp_objects_output_dir, exist_ok=True)
        for tirp in tirps_to_save:
            try:
                # Assume save_tirp_object exists and works
                tirp.save_tirp_object(tirp_objects_output_dir)
                saved_count += 1
            except Exception as e:
                tirp_str = getattr(tirp, 'to_string', lambda: 'UnknownTIRP')()
                print(f"Warning: Failed to save TIRP object {tirp_str}: {e}")

    print(f"Saved {saved_count} unique selected TIRP objects to {tirp_objects_output_dir}")

    end_total_time = time.time()
    print(f"--- Finished Stage 2: TIRP Mining. Total Time: {end_total_time - start_total_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Run data splitting and KarmaLego TIRP mining (Simplified).")

    parser.add_argument("--abstraction_output_dir", required=True)
    parser.add_argument("--mining_run_dir", required=True)
    parser.add_argument("--tirp_objects_output_dir", required=True)
    parser.add_argument("--mvs", required=True, type=float)
    parser.add_argument("--max_gap", required=True, type=int)
    parser.add_argument("--relations", required=True, type=int)
    parser.add_argument("--skip_followers", required=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--epsilon", required=True, type=int)
    parser.add_argument("--event_symbol", required=True)

    args = parser.parse_args()

    run_single_mining_task(
        abstraction_output_dir=args.abstraction_output_dir,
        mining_run_dir=args.mining_run_dir,
        tirp_objects_output_dir=args.tirp_objects_output_dir,
        mvs=args.mvs,
        max_gap=args.max_gap,
        relations=args.relations,
        skip_followers=args.skip_followers,
        epsilon=args.epsilon,
        event_symbol=args.event_symbol
    )

    sys.exit(0)