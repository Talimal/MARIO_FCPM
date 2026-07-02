import os
import argparse
import sys
import pandas as pd
import logging
import gc
import traceback
import hashlib

# Importing existing functions from your pipeline
from FCPM_Package.Continuous_Prediction import predict_continuous
from FCPM_Package.Evaluate import Evaluate

try:
    from config import SUPPORTING_ENTITIES_ONLY
except ImportError:
    logging.warning("SUPPORTING_ENTITIES_ONLY not found in config. Defaulting to False (all entities).")
    SUPPORTING_ENTITIES_ONLY = False

# Configure logging
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

def _get_train_entities(class_file_path):
    """ Reads the entity-class relations file and returns a list of all entities. """
    class_df = pd.read_csv(class_file_path, dtype={"EntityID": int, "ClassID": int})
    return class_df["EntityID"].tolist()

def _extract_and_return_summary_metrics(roc_auc, auprc, detailed_csv_path, tirp_name):
    """
    Extracts the best metrics specifically for the FCPM validation.
    Ensures the chosen threshold is > 0.0. If no valid threshold exists, 
    defaults to 1.0 ("always say no") and 0.0 for metrics.
    """
    try:
        details_df = pd.read_csv(detailed_csv_path)
        
        # Filter out the threshold 0.0 to avoid the trivial "predict all positive" edge case
        valid_df = details_df[details_df['threshold'] > 0.0].copy()
        
        if valid_df.empty or valid_df['F1'].isna().all() or (valid_df['F1'] == 0).all():
            optimal_threshold = 1.0 
            max_f1, max_precision, recall_at_f1, acc_at_f1 = 0.0, 0.0, 0.0, 0.0
            tp, fp, tn, fn = 0, 0, 0, 0
        else:
            best_f1_idx = valid_df['F1'].idxmax()
            best_f1_row = valid_df.loc[best_f1_idx]
            
            max_precision = valid_df.loc[best_f1_idx, 'Precision']
            
            optimal_threshold = best_f1_row['threshold']
            max_f1 = best_f1_row['F1']
            recall_at_f1 = best_f1_row['Recall']
            acc_at_f1 = best_f1_row['Accuracy']
            tp = best_f1_row['TP']
            fp = best_f1_row['FP']
            tn = best_f1_row['TN']
            fn = best_f1_row['FN']
        
        # Build the summary dictionary using the exact requested column names
        return {
            "TIRP_name": tirp_name,
            "val_AUC": roc_auc,
            "val_AUPRC": auprc,
            "val_Precision": max_precision,
            "val_F1": max_f1,
            "Optimal_Threshold": optimal_threshold,
            "Train_Recall": recall_at_f1,
            "Train_Accuracy": acc_at_f1,
            "Train_TP": tp,
            "Train_FP": fp,
            "Train_TN": tn,
            "Train_FN": fn
        }
        
    except Exception as e:
        logging.error(f"Failed to extract summary metrics: {e}")
        return {}

def run_train_validation_for_tirp(tirp_path, tirp_model_run_dir, train_class_file_path, epsilon, ew_window_size, ew_early_warning_value):
    """
    Main validation function for a single TIRP.
    Runs prediction using pre-built inference tables on the FCPM model,
    and aggregates the results into a single CSV.
    """
    sanitized_id = get_sanitized_tirp_id(tirp_path)
    tirp_output_dir = os.path.join(tirp_model_run_dir, f'tirp_{sanitized_id}')
    
    # 1. Fault Tolerance: Check if already validated
    done_file_path = os.path.join(tirp_output_dir, f'stage3_val_{sanitized_id}.done')
    if os.path.exists(done_file_path):
        logging.info(f"Skipping TIRP {sanitized_id} (Already Validated: {done_file_path})")
        return True

    logging.info(f"Starting FCPM Train Validation for TIRP: {sanitized_id}")

    # 2. Locate pre-built inference tables from Stage 3
    train_inference_dir = os.path.join(tirp_output_dir, "train_inference")
    duration_file = os.path.join(train_inference_dir, "durations_merged_df.csv")
    event_dict_file = os.path.join(train_inference_dir, "event_time_dict.pkl")

    if not os.path.exists(duration_file) or not os.path.exists(event_dict_file):
        logging.warning(f"Inference tables missing for {sanitized_id}. Cannot validate.")
        return False

    all_train_entities = _get_train_entities(train_class_file_path)  
    # Filter entities based on config flag
    if SUPPORTING_ENTITIES_ONLY:
        # The durations file contains only entities where the TIRP actually started
        durations_df = pd.read_csv(duration_file, usecols=['EntityID'])
        supporting_entities = durations_df['EntityID'].unique().tolist()
        
        # Intersect with train entities to ensure consistency
        entities_to_evaluate = list(set(all_train_entities).intersection(set(supporting_entities)))
        logging.info(f"Config set to conditional validation: Using {len(entities_to_evaluate)} supporting entities out of {len(all_train_entities)} total.")
        
        if not entities_to_evaluate:
            logging.warning(f"TIRP {sanitized_id} has no supporting train entities. Aborting validation.")
            return True
    else:
        entities_to_evaluate = all_train_entities
        logging.info(f"Config set to global validation: Using all {len(entities_to_evaluate)} train entities.")
        
    
    # 3. Check if FCPM models exist
    fcpm_model_path = os.path.join(tirp_output_dir, 'models', f"{sanitized_id}-FCPM.pkl")
    tte_model_path = os.path.join(tirp_output_dir, 'models', f"{sanitized_id}-TTE.pkl")

    if not os.path.exists(fcpm_model_path) or not os.path.exists(tte_model_path):
        logging.warning(f"FCPM/TTE trained models missing for {sanitized_id}. Cannot validate.")
        return False

    train_predictions_dir = os.path.join(tirp_output_dir, "train_predictions")
    # os.makedirs(train_predictions_dir, exist_ok=True)

    # 4. Construct the mock models_path dictionary for predict_continuous
    mock_models_path = {
        sanitized_id: {
            "FCPM": fcpm_model_path, 
            "TTE": tte_model_path,
            "duration_test_file": duration_file,
            "event_time_dict": event_dict_file
        }
    }

    try:
        # 5. Predict Continuous
        print(f"Predicting continuous values for {sanitized_id}...")
        predict_continuous(
            base_dir=tirp_output_dir,
            output_dir=train_predictions_dir,
            class_file=train_class_file_path,
            entities_list=entities_to_evaluate,
            epsilon=epsilon,
            models_path=mock_models_path
        )

        print(f"Aggregating continuous values for {sanitized_id}...")
        # 5.5 Aggregate multiple instances of the same TIRP per time unit (TFS)
        # Using the standard Aggregation class for consistency
        from FCPM_Package.aggregation_class import Aggregation
        aggregator = Aggregation(
            method_name="avg",
            prediction_output_dir=f"{train_predictions_dir}_FCPM",
            tirp_selection_method=None,
            scores_file_path=None
        )
        agg_file_name = aggregator.aggregate_predictions(entity_ids=entities_to_evaluate)

        # 6. Evaluate
        if not agg_file_name:
            logging.error(f"Aggregation failed to produce an output file for {sanitized_id}")
            return False

        print(f"Evaluating continuous values for {sanitized_id}...")
        evaluator = Evaluate(
            prediction_output_dir=f"{train_predictions_dir}_FCPM",
            agg_method="avg",
            agg_file_name=agg_file_name,
            save_threshold_files=False
        )

        detailed_metrics_csv = os.path.join(tirp_output_dir, "train_metrics_all_thresholds.csv")
        roc_auc, auprc = evaluator.evaluate_roc(tte_w_list=[ew_window_size], e_w_list=[ew_early_warning_value],val_output_csv=detailed_metrics_csv)

        print(f"Extracting metrics for {sanitized_id}... ROC-AUC: {roc_auc}, AUPRC: {auprc}")
        # 7. Extract metrics and save to CSV
        summary_dict = _extract_and_return_summary_metrics(roc_auc, auprc, detailed_metrics_csv, sanitized_id)
        
        if summary_dict:
            summary_df = pd.DataFrame([summary_dict])
            summary_file_path = os.path.join(tirp_output_dir, "train_summary_metrics.csv")
            summary_df.to_csv(summary_file_path, index=False)
            logging.info(f"Successfully saved train summary metrics for TIRP {sanitized_id}")

        # 8. Create Done File
        with open(done_file_path, 'w') as f_done:
            f_done.write("done")

    except Exception as e:
        logging.error(f"Error during validation for {sanitized_id}: {e}")
        traceback.print_exc()
        return False

    finally:
        # Cleanup memory
        gc.collect()

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3.5: Train Validation for FCPM models.")
    
    parser.add_argument("--tirp_model_run_dir", required=True, help="Base directory (feature_matrix_dir).")
    parser.add_argument("--tirp_list_file", required=True, help="Path to text file containing list of TIRP .pkl paths.")
    parser.add_argument("--abstraction_output_dir", required=True, help="Stage 1 output dir (for entity-class-relations.csv).")
    parser.add_argument("--epsilon", required=True, type=float, help="Epsilon parameter.")
    parser.add_argument("--ew_window_size", required=True, type=float, help="Window size scalar")
    parser.add_argument("--ew_early_warning_value", required=True, type=int, help="Consecutive threshold triggers required")

    args = parser.parse_args()

    train_class_file_path = os.path.join(args.abstraction_output_dir, 'Train', 'entity-class-relations.csv')

    if not os.path.exists(args.tirp_list_file):
        sys.stderr.write(f"ERROR: TIRP list file not found: {args.tirp_list_file}\n")
        sys.exit(1)
        
    with open(args.tirp_list_file, 'r') as f:
        tirp_paths = [line.strip() for line in f if line.strip()]
        
    logging.info(f"Found {len(tirp_paths)} TIRPs for validation.")

    tirp_val_success = True

    for tirp_path in tirp_paths:
        success = run_train_validation_for_tirp(
            tirp_path=tirp_path,
            tirp_model_run_dir=args.tirp_model_run_dir,
            train_class_file_path=train_class_file_path,
            epsilon=args.epsilon,
            ew_window_size=args.ew_window_size,
            ew_early_warning_value=args.ew_early_warning_value
        )
        if not success:
            tirp_val_success = False
    
    if tirp_val_success:
        logging.info("Stage 3.5 completed successfully for all TIRPs in the batch.")
        sys.exit(0)
    else:
        logging.error("Stage 3.5 failed for at least one TIRP in the batch.")
        sys.exit(1)


