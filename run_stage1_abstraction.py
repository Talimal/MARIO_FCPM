# Filename: run_stage1_abstraction.py

import os
import argparse
import sys
import time
import json
import pandas as pd

# Import configuration
from config import GRADIENT_WINDOW_SIZE, KB_STATES_PATH, GRADIENT_CUTOFFS_PATH, SUPERVISED_ABSTRACTION_METHODS

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


def is_supervised_method(method):
    """
    True if `method` requires class labels (and is therefore invalid for MARIO,
    which has no classes). Covers the configured supervised list plus the tid3/
    td4c/mdlp families by prefix.
    """
    m = method.lower()
    if method in SUPERVISED_ABSTRACTION_METHODS:
        return True
    return m.startswith("td4c") or m.startswith("tid3") or m == "mdlp"


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

        else:
            # Unsupervised methods (equal_frequency, equal_width, sax, persist, ...)
            method_config["default"].append({
                "method": method,
                "bins": num_of_bins
            })

    return method_config


def run_single_abstraction(abstraction_output_dir, train_data_file, test_data_file,
                          d_method, num_of_bins, interpolation_gap):
    """
    Runs Hugobot abstraction (learn states on train, apply to test) for a single
    parameter combination, for MARIO forecasting.

    No class/event handling: MARIO has no classes, so supervised methods and the
    event-based split are removed. Hugobot always writes a single 'KL.txt' per
    split (all entities); the per-class 'KL-class-*.txt' files are only produced
    when class labels are present and are ignored downstream.

    Parameters:
        abstraction_output_dir (str): Base directory where 'Train' and 'Test'
                                      subdirectories will be created.
        train_data_file (str): Path to the split's train.csv.
        test_data_file (str): Path to the split's test.csv.
        d_method (str): Unsupervised discretization method (e.g. "equal_frequency").
        num_of_bins (int): Number of bins for discretization.
        interpolation_gap (int): Maximum allowed gap (passed as 'max_gap' to Hugobot).

    Note: KB and gradient configuration is loaded from config.py
        (GRADIENT_WINDOW_SIZE, KB_STATES_PATH, GRADIENT_CUTOFFS_PATH).
    """
    print("--- Starting Stage 1: Abstraction (MARIO forecasting) ---")
    print(f"Parameters: d_method={d_method}, num_of_bins={num_of_bins}, interpolation_gap={interpolation_gap}")
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

    # MARIO forbids supervised abstraction (no class labels exist).
    supervised = [m for m in methods if is_supervised_method(m)]
    if supervised:
        print(f"ERROR: Supervised abstraction method(s) {supervised} are not supported in MARIO "
              f"(forecasting has no class labels). Use unsupervised methods "
              f"(equal_frequency, equal_width, sax, persist, gradient, knowledge/kb_gradient).")
        sys.exit(1)

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
        print(f"  Abstraction finished. Time: {time.time() - start_time:.2f} seconds.")

    else:
        # Composite mode or single non-knowledge (unsupervised) method
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

        # Train phase: learn states on the train slice and save.
        print("\nRunning Hugobot abstraction on training data...")
        start_time = time.time()
        train_df = pd.read_csv(train_data_file, low_memory=False)

        train_ta = TemporalAbstraction(train_df)
        _, train_states = train_ta.apply(
            method_config=method_config,
            per_variable=True,
            split_test=False,
            save_output=True,
            output_dir=os.path.join(abstraction_output_dir, "Train"),
            max_gap=interpolation_gap
        )
        print(f"  Training abstraction finished. Time: {time.time() - start_time:.2f} seconds.")

        # Test phase: apply the learned states to the test data.
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
        print(f"  Test abstraction finished. Time: {time.time() - start_time:.2f} seconds.")

    # --- Summary visualization (non-fatal): raw signal colored by learned StateID. ---
    try:
        from pipeline_viz import save_stage1_abstraction_example
        save_stage1_abstraction_example(abstraction_output_dir, d_method=d_method, num_of_bins=num_of_bins)
    except Exception as e:
        print(f"[viz] Stage 1 visualization skipped (non-fatal): {e}")

    print("\n--- Finished Stage 1: Abstraction ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 (MARIO): unsupervised Hugobot abstraction on train/test for one parameter combination.")

    # Arguments provided by the orchestrator (submit_stage1_job)
    parser.add_argument("--train_data_file", required=True, help="Path to the split's train.csv.")
    parser.add_argument("--test_data_file", required=True, help="Path to the split's test.csv.")
    parser.add_argument("--d_method", required=True, help="Unsupervised discretization method (e.g. 'equal_frequency', 'sax').")
    parser.add_argument("--num_of_bins", required=True, type=int, help="Number of bins for discretization.")
    parser.add_argument("--interpolation_gap", required=True, type=int, help="Maximum allowed gap for interpolation.")
    parser.add_argument("--abstraction_output_dir", required=True, help="Base directory where 'Train' and 'Test' subdirectories will be created.")

    args = parser.parse_args()

    run_single_abstraction(
        abstraction_output_dir=args.abstraction_output_dir,
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        d_method=args.d_method,
        num_of_bins=args.num_of_bins,
        interpolation_gap=args.interpolation_gap
    )

    # Exit successfully (sbatch script will create the .done file)
    sys.exit(0)
