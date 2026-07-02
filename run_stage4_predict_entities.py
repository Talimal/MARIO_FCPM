import os
import argparse
import sys
from config import EPSILON_FCPM
from FCPM_Package.Continuous_Prediction import *
from pathlib import Path



def run_prediction_for_batch(entity_list_file, built_models_base_dir,
                             prediction_batch_output_dir, epsilon):
    """
    Runs prediction for a specific batch of entities.
    (Simplified - assumes valid inputs and successful execution).
    """
    # --- Step 1: Read entity list for this batch ---
    # Assumes file exists and contains valid entity IDs (one per line)
    # Errors during file open/read will cause script to fail naturally.
    with open(entity_list_file, 'r') as f:
        entities_list = [int(line.strip()) for line in f if line.strip()]

        # build class‑file path
    base_dir = Path(built_models_base_dir)
    # “two folders back”  →  parent.parent
    class_file_path = base_dir.parent.parent / "Test" / "entity-class-relations.csv"

    if not class_file_path.is_file():
        raise FileNotFoundError(f"Class‑file not found: {class_file_path}")

    predict_continuous(
        base_dir=built_models_base_dir,              # First arg -> Model location
        output_dir=prediction_batch_output_dir, # Second arg -> Output location for this batch
        entities_list=entities_list,               # List of entities for this batch
        epsilon=epsilon,                           # Epsilon value
        class_file=str(class_file_path)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Run prediction for a batch of entities.")

    # Removed test_data_path as it wasn't used in the simplified run_prediction_for_batch
    # Add it back if your predict_continuous function needs it.
    # parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--entity_list_file", required=True)
    parser.add_argument("--built_models_base_dir", required=True)
    parser.add_argument("--prediction_output_dir", required=True)

    args = parser.parse_args()

    # Directly call the processing function
    # Pass test_data_path=args.test_data_path if needed by your function
    run_prediction_for_batch(
        entity_list_file=args.entity_list_file,
        built_models_base_dir=args.built_models_base_dir,
        prediction_batch_output_dir=args.prediction_output_dir,
        epsilon = EPSILON_FCPM
    )

    # Exit successfully (sbatch script will create the .done file)
    sys.exit(0)