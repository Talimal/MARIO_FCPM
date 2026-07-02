# Filename: run_stage1_abstraction.py

import os
import argparse
import sys
import time
import json
import pandas as pd

# Import configuration
from config import GRADIENT_WINDOW_SIZE, KB_STATES_PATH, GRADIENT_CUTOFFS_PATH

# Try to import Hugobot functions. Handle import error gracefully.
try:
    from Hugobot2.ta_package import TemporalAbstraction
    from Hugobot2.ta_package import utils
except ImportError:
    print("ERROR: Could not import TemporalAbstraction from Hugobot2.ta_package.")


def load_gradient_cutoffs(gradient_cutoffs_path):
    """
    Load gradient cutoffs from a JSON file.
    
    Expected format:
    {
        "cutoffs": {
            "1": [-20, 20],
            "2": [-30, 30],
            ...
        },
        "default": [-30, 30]
    }
    
    Returns a dictionary mapping variable IDs (as integers) to cutoff lists.
    """
    if not gradient_cutoffs_path or not os.path.exists(gradient_cutoffs_path):
        return None
    
    with open(gradient_cutoffs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert string keys to integers for variable IDs
    cutoffs = data.get("cutoffs", {})
    gradient_cutoffs = {int(k): v for k, v in cutoffs.items()}
    
    # Add default if present
    if "default" in data:
        gradient_cutoffs["default"] = data["default"]
    
    return gradient_cutoffs


def parse_method_string(d_method):
    """
    Parse a method string into a list of individual methods.
    
    Examples:
        "sax" -> ["sax"]
        "equal_frequency_and_gradient" -> ["equal_frequency", "gradient"]
        "knowledge_and_kb_gradient_and_sax" -> ["knowledge", "kb_gradient", "sax"]
    
    Returns:
        List of method names
    """
    if "_and_" in d_method:
        return d_method.split("_and_")
    else:
        return [d_method]


def build_method_config(methods, num_of_bins, gradient_window_size, 
                       kb_states_path=None, gradient_cutoffs_path=None):
    """
    Build a method_config dictionary for composite mode based on parsed methods.
    
    Parameters:
        methods (list): List of method names (e.g., ["knowledge", "gradient", "sax"])
        num_of_bins (int): Number of bins for discretization methods
        gradient_window_size (int): Window size for gradient methods
        kb_states_path (str): Path to KB states CSV file
        gradient_cutoffs_path (str): Path to gradient cutoffs JSON file
    
    Returns:
        dict: method_config in the format expected by TemporalAbstraction
    """
    method_config = {"default": []}
    
    # Load KB states if needed
    kb_states = None
    if "knowledge" in methods and kb_states_path and os.path.exists(kb_states_path):
        kb_states = pd.read_csv(kb_states_path)
        print(f"  Loaded KB states from: {kb_states_path}")
    
    # Load gradient cutoffs if needed
    gradient_cutoffs = None
    if ("kb_gradient" in methods or "gradient" in methods) and gradient_cutoffs_path:
        gradient_cutoffs = load_gradient_cutoffs(gradient_cutoffs_path)
        if gradient_cutoffs:
            print(f"  Loaded gradient cutoffs from: {gradient_cutoffs_path}")
    
    # Build configuration for each method
    for method in methods:
        if method == "knowledge":
            # Knowledge-based method
            if kb_states is not None:
                method_config["default"].append({
                    "method": "knowledge",
                    "states": kb_states
                })
            else:
                print(f"  WARNING: 'knowledge' method specified but KB states not found at {kb_states_path}")
                print("           Skipping knowledge method.")
        
        elif method == "kb_gradient":
            # Knowledge-based gradient
            if gradient_cutoffs is not None:
                method_config["default"].append({
                    "method": "gradient",
                    "gradient_window_size": gradient_window_size,
                    "sub_method": "knowledge",
                    "knowledge_cutoffs": gradient_cutoffs,
                    "bins": 3  # KB gradient typically uses 3 states
                })
            else:
                print("  WARNING: 'kb_gradient' method specified but gradient cutoffs not found")
                print("           Falling back to quantile-based gradient.")
                method_config["default"].append({
                    "method": "gradient",
                    "gradient_window_size": gradient_window_size,
                    "sub_method": "quantile",
                    "bins": 3
                })
        
        elif method == "gradient":
            # Regular gradient (quantile-based)
            method_config["default"].append({
                "method": "gradient",
                "gradient_window_size": gradient_window_size,
                "sub_method": "quantile",
                "bins": num_of_bins
            })
        elif method.startswith("tid3"):
            base_method, min_mean_gap_override = utils.strip_mingap_suffix(method)
            (scoring_method, duration_preference, multivariate_refinement,
             nb_candidates, mv_variable_order, mv_random_seed) = utils.parse_tid3_config(base_method)
            tid3_cfg = {
                "method": "tid3",
                "bins": num_of_bins,
                "scoring_method": scoring_method,
                "duration_preference": duration_preference,
                "nb_candidates": nb_candidates if nb_candidates is not None else 150,
                "multivariate_refinement": multivariate_refinement,
                "mv_variable_order": mv_variable_order,
                "fallback_method": "td4c",
                "fallback_td4c_distance": "cosine",
            }
            if mv_variable_order == "random":
                if mv_random_seed is None:
                    raise ValueError(
                        f"Random ordering requires an explicit seed suffix, e.g. 'tid3_mv_random_seed0'. "
                        f"Got d_method='{method}'."
                    )
                tid3_cfg["mv_random_seed"] = mv_random_seed
            if min_mean_gap_override is not None:
                tid3_cfg["min_mean_gap"] = min_mean_gap_override
            method_config["default"].append(tid3_cfg)
            mingap_display = min_mean_gap_override if min_mean_gap_override is not None else "default(0.0)"
            seed_display = mv_random_seed if mv_variable_order == "random" else "n/a"
            print(f"  TID3 config: scoring={scoring_method}, duration_pref={duration_preference}, multivariate={multivariate_refinement}, mv_variable_order={mv_variable_order}, mv_random_seed={seed_display}, min_mean_gap={mingap_display}, fallback=td4c/cosine")

        elif method.startswith("td4c"):
            # TD4C method with optional configuration
            # Parse distance measure from method name
            distance_measure = utils.parse_td4c_config(method)
            method_config["default"].append({
                "method": "td4c",
                "bins": num_of_bins,
                "distance_measure": distance_measure
            })
            print(f"  TD4C config: distance_measure={distance_measure}, bins={num_of_bins}")

        elif method == "mdlp":
            method_config["default"].append({
                "method": "mdlp",
                "bins": num_of_bins,
            })
            print(f"  MDLP config: bins={num_of_bins}, per_variable=True")

        else:
            # Other methods (equal_frequency, equal_width, sax, td4c, etc.)
            method_config["default"].append({
                "method": method,
                "bins": num_of_bins
            })
    
    return method_config


def prepare_event_based_split(df, event_window):
    """
    Splits the dataframe based on event logical classes using vectorized operations.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        event_window (str/int/float): Window parameter.
        
    Returns:
        pd.DataFrame: Processed dataframe with split entities.
    """
    print(f"  Running prepare_event_based_split with event_window={event_window} (Vectorized)...")
    
    # helper to parse window
    try:
        w_val = float(event_window)
    except (ValueError, TypeError):
        print(f"    ERROR: Invalid event_window value: {event_window}. Returning original.")
        return df

    is_percentage = 0.0 < w_val < 1.0

    # Ensure EntityID is string for suffixing operations
    # operate on a copy to avoid SettingWithCopy warnings on the original df if it was a view
    df = df.copy() 
    df['EntityID'] = df['EntityID'].astype(str)

    # 1. Identify Classes
    # Filter for class definition rows (TemporalPropertyID == -1)
    class_rows_mask = df['TemporalPropertyID'] == -1
    class_df = df[class_rows_mask]
    
    # Map EntityID -> Class Value
    # We want entities where Class Value != 0
    # Create a set of IDs that are Class 1
    class_1_ids = set(class_df[class_df['TemporalPropertyValue'] != 0]['EntityID'])
    
    if not class_1_ids:
        print("    No Class 1 entities found to split.")
        return df 
        
    # Mask for all rows belonging to Class 1 entities
    is_class_1 = df['EntityID'].isin(class_1_ids)
    
    # Split main DataFrame into Class 0 (untouched) and Class 1 (to be processed)
    df_class_0 = df[~is_class_1] 
    df_class_1 = df[is_class_1]
    
    # 2. Process Class 1 Data
    # Separate data and class rows for Class 1 subset
    c1_meta_mask = df_class_1['TemporalPropertyID'] == -1
    df_c1_data = df_class_1[~c1_meta_mask].copy()
    df_c1_meta = df_class_1[c1_meta_mask].copy()
    
    if df_c1_data.empty:
         return df # Should not happen if class rows exist, but safety check

    # Calculate stats per entity: min and max timestamp
    stats = df_c1_data.groupby('EntityID')['TimeStamp'].agg(['min', 'max'])
    
    # Merge stats back to data to allow row-wise vector operations
    df_c1_data = df_c1_data.merge(stats, left_on='EntityID', right_index=True)
    
    # Calculate Split Threshold per row
    duration = df_c1_data['max'] - df_c1_data['min']
    
    if is_percentage:
         split_val = duration * w_val
    else:
         split_val = w_val
         
    # Split point: Time > (Max - Window) -> Class 1 (Near Event)
    split_threshold = df_c1_data['max'] - split_val
    
    mask_high = df_c1_data['TimeStamp'] > split_threshold
    mask_low = ~mask_high  # data <= split_threshold (Class 0 / Distant)
    
    # Create new splits
    
    # --- Part 1 (High / Near Event) -> Becomes Class 1 ---
    df_part1 = df_c1_data[mask_high].copy()
    if not df_part1.empty:
        df_part1['EntityID'] = df_part1['EntityID'] + '_1'
    
    # --- Part 0 (Low / Distant) -> Becomes Class 0 ---
    df_part0 = df_c1_data[mask_low].copy()
    if not df_part0.empty:
        df_part0['EntityID'] = df_part0['EntityID'] + '_0'
    
    # Remove the temporary 'min' and 'max' columns derived from merge
    df_part1.drop(columns=['min', 'max'], inplace=True, errors='ignore')
    df_part0.drop(columns=['min', 'max'], inplace=True, errors='ignore')
    
    # 3. Create Class Rows for the new entities
    # We filter to ensure we only create metadata for entities that actually have data in that split
    
    ids_in_part1 = set(df_part1['EntityID'].unique())
    ids_in_part0 = set(df_part0['EntityID'].unique())
    
    # Meta for Part 1 (Class 1)
    meta_1 = df_c1_meta.copy()
    meta_1['EntityID'] = meta_1['EntityID'] + '_1'
    meta_1['TemporalPropertyValue'] = 1.0 # Force Class 1
    meta_1 = meta_1[meta_1['EntityID'].isin(ids_in_part1)]
    
    # Meta for Part 0 (Class 0)
    meta_0 = df_c1_meta.copy()
    meta_0['EntityID'] = meta_0['EntityID'] + '_0'
    meta_0['TemporalPropertyValue'] = 0.0 # Force Class 0
    meta_0 = meta_0[meta_0['EntityID'].isin(ids_in_part0)]
    
    # 4. Concatenate All Parts
    frames = [df_class_0, df_part0, df_part1, meta_0, meta_1]
    # Filter out empty frames
    frames = [f for f in frames if not f.empty]
    
    if not frames:
        return pd.DataFrame(columns=df.columns)

    result_df = pd.concat(frames, ignore_index=True)
    
    # 5. Final Sort (Recommended to keep data ordered)
    result_df.sort_values(by=['EntityID', 'TimeStamp'], inplace=True)
    
    # Reset index
    result_df.reset_index(drop=True, inplace=True)
    
    print(f"    Split complete via Vectorization. Original entities: {df['EntityID'].nunique()}, New entities: {result_df['EntityID'].nunique()}")
    
    return result_df


def run_single_abstraction(abstraction_output_dir, train_data_file, test_data_file, 
                          d_method, num_of_bins, interpolation_gap,
                          split_event_class=False, event_window=None):
    """
    Runs Hugobot abstraction (training and knowledge-based testing) for a single
    set of parameters and a specific train/test split.

    Parameters:
        abstraction_output_dir (str): The base directory where 'Train' and 'Test'
                                      subdirectories for this run will be created/used.
        train_data_file (str): Path to the input train.csv file.
        test_data_file (str): Path to the input test.csv file.
        d_method (str): Discretization method (e.g., "sax", "knowledge_and_kb_gradient").
        num_of_bins (int): Number of bins for discretization.
        interpolation_gap (int): Maximum allowed gap for interpolation. This is passed
                                 as 'max_gap' to the Hugobot functions.
    
    Note: KB and gradient configuration is loaded from config.py:
        - GRADIENT_WINDOW_SIZE
        - KB_STATES_PATH
        - GRADIENT_CUTOFFS_PATH
    """
    print("--- Starting Stage 1: Abstraction ---")
    print(f"Parameters: d_method={d_method}, num_of_bins={num_of_bins}, interpolation_gap={interpolation_gap}")
    print(f"            split_event_class={split_event_class}, event_window={event_window}")
    print("Static Config (from config.py):")
    print(f"  gradient_window_size={GRADIENT_WINDOW_SIZE}")
    print(f"  kb_states_path={KB_STATES_PATH}")
    print(f"  gradient_cutoffs_path={GRADIENT_CUTOFFS_PATH}")
    print(f"Input Train Data: {train_data_file}")
    print(f"Input Test Data: {test_data_file}")
    print(f"Output Directory: {abstraction_output_dir}")

    # --- Input Validation ---
    if not os.path.exists(train_data_file):
        print(f"ERROR: Training data file not found at {train_data_file}")
        sys.exit(1)
    if not os.path.exists(test_data_file):
        print(f"ERROR: Test data file not found at {test_data_file}")
        sys.exit(1)

    # Parse the method string into individual methods
    methods = parse_method_string(d_method)
    print(f"\nParsed methods: {methods}")
    
    # Check if this is a pure knowledge-based approach (single method)
    if len(methods) == 1 and methods[0] == "knowledge":
        # Original knowledge-based single-method approach
        if not KB_STATES_PATH or not os.path.exists(KB_STATES_PATH):
            print("ERROR: Knowledge method requires kb_states_path")
            sys.exit(1)
            
        states = pd.read_csv(KB_STATES_PATH)
        train_df = pd.read_csv(train_data_file, low_memory=False)
        test_df = pd.read_csv(test_data_file, low_memory=False)
        
        ta_train = TemporalAbstraction(train_df)
        ta_test = TemporalAbstraction(test_df)
        
        start_time = time.time()
        _, _ = ta_train.apply(
            method="knowledge", 
            train_states=states, 
            output_dir=os.path.join(abstraction_output_dir, "Train"), 
            max_gap=interpolation_gap
        )
        _, _ = ta_test.apply(
            method="knowledge", 
            train_states=states, 
            output_dir=os.path.join(abstraction_output_dir, "Test"), 
            max_gap=interpolation_gap
        )
        end_time = time.time()
        print(f"  Abstraction finished. Time: {end_time - start_time:.2f} seconds.")
    
    else:
        # Composite mode or single non-knowledge method
        method_config = build_method_config(
            methods=methods,
            num_of_bins=num_of_bins,
            gradient_window_size=GRADIENT_WINDOW_SIZE,
            kb_states_path=KB_STATES_PATH,
            gradient_cutoffs_path=GRADIENT_CUTOFFS_PATH
        )
        
        print("\nBuilt method_config:")
        for i, cfg in enumerate(method_config["default"]):
            print(f"  Method {i+1}: {cfg}")

        # Train phase
        print("\nRunning Hugobot abstraction on training data...")
        start_time = time.time()
        train_df = pd.read_csv(train_data_file, low_memory=False)
        
        train_states = None

        if split_event_class:
            print(f"  --- Split Event Class Mode Enabled (Window: {event_window}) ---")
            
            # 1. Prepare split data (Learn states from this)
            print("  1. Preparing split data for state learning...")
            split_train_df = prepare_event_based_split(train_df, event_window)
            
            # 2. Learn states (Training on split data, NOT saving output)
            print("  2. Learning states from split data (save_output=False)...")
            split_ta = TemporalAbstraction(split_train_df)
            _, train_states = split_ta.apply(
                method_config=method_config,
                per_variable=True,
                split_test=False,
                save_output=False, # Logic: Don't save these abstractions
                output_dir=None,
                max_gap=interpolation_gap
            )
            
            # 3. Apply states to ORIGINAL Training Data (Save output here)
            print("  3. Applying learned states to ORIGINAL Training Data...")
            # We use map_states_to_test_composite to apply states to a dataframe
            utils.map_states_to_test_composite(
                test_df=train_df, # Using train_df as target
                states_list=train_states,
                method_config=method_config,
                output_dir=os.path.join(abstraction_output_dir, "Train"),
                max_gap=interpolation_gap
            )

        else:
            # Standard Mode: Train and save on original data
            print("  --- Standard Mode ---")
            train_ta = TemporalAbstraction(train_df)

            _, train_states = train_ta.apply(
                method_config=method_config,
                per_variable=True,
                split_test=False,
                save_output=True,
                output_dir=os.path.join(abstraction_output_dir, "Train"),
                max_gap=interpolation_gap
            )
            
        end_time = time.time()
        print(f"  Training abstraction finished. Time: {end_time - start_time:.2f} seconds.")

        # Test phase
        print("\nApplying learned states to test data...")
        test_df = pd.read_csv(test_data_file, low_memory=False)
        start_time = time.time()

        utils.map_states_to_test_composite(
            test_df=test_df, 
            states_list=train_states,
            method_config=method_config, 
            output_dir=os.path.join(abstraction_output_dir, "Test"),
            max_gap=interpolation_gap
        )

        end_time = time.time()
        print(f"  Test abstraction finished. Time: {end_time - start_time:.2f} seconds.")
    
    print("\n--- Finished Stage 1: Abstraction ---")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Run Hugobot abstraction on train and test data for a specific parameter combination.")

    # Define command-line arguments expected from the orchestrator (submit_stage1_job)
    parser.add_argument("--train_data_file", required=True, help="Path to the input train.csv file for the current fold.")
    parser.add_argument("--test_data_file", required=True, help="Path to the input test.csv file for the current fold.")
    parser.add_argument("--d_method", required=True, help="Discretization method (e.g., 'sax', 'knowledge_and_kb_gradient').")
    parser.add_argument("--num_of_bins", required=True, type=int, help="Number of bins for discretization.")
    parser.add_argument("--interpolation_gap", required=True, type=int, help="Maximum allowed gap for interpolation.")
    parser.add_argument("--abstraction_output_dir", required=True, help="Base directory where 'Train' and 'Test' subdirectories will be created/used for this run's output.")
    
    # New arguments for Supervised Abstraction
    parser.add_argument("--split_event_class", type=str, default="False", help="Whether to split data based on event class (True/False).")
    parser.add_argument("--event_window", type=str, default=None, help="Window size for event/class split (float for percentage, int for time units).")
    
    # These parameters are now static/hardcoded in the function, but kept for backward compatibility
    parser.add_argument("--gradient_window_size", type=int, default=120, help="(Ignored - now hardcoded to 120)")
    parser.add_argument("--kb_states_path", type=str, default=None, help="(Ignored - now hardcoded)")
    parser.add_argument("--gradient_cutoffs_path", type=str, default=None, help="(Ignored - now hardcoded)")

    args = parser.parse_args()

    # Call the main processing function with parsed arguments
    # Note: gradient_window_size, kb_states_path, and gradient_cutoffs_path are now hardcoded in the function
    run_single_abstraction(
        abstraction_output_dir=args.abstraction_output_dir,
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        d_method=args.d_method,
        num_of_bins=args.num_of_bins,
        interpolation_gap=args.interpolation_gap,
        split_event_class=(str(args.split_event_class).lower() == 'true'), # Convert string to boolean
        event_window=args.event_window
    )

    # Exit successfully (sbatch script will create the .done file)
    sys.exit(0)