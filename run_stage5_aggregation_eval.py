import os
import argparse
import sys
import time
import csv # For CSV file operations

# --- Assume necessary imports succeed ---
# Adjust paths based on your project structure
try:
    from FCPM_Package.aggregation_class import Aggregation
    from FCPM_Package.Evaluate import Evaluate
except ImportError as e:
    print(f"ERROR: Failed to import Aggregation or Evaluate from FCPM_Package: {e}")
    print("Please ensure FCPM_Package is installed and accessible.")
    sys.exit(1)

# --- Main Aggregation & Evaluation Function (Simplified) ---

def run_single_aggregation_evaluation(
    aggregation_method, num_tirps_for_selection, tirp_selection_method, 
    tte_w_list, e_w_list,
    tirp_models_base_dir, results_output_dir, # Directory for the ROC CSVs of this specific run
    dataset_name, fold_num, abs_d_method, abs_b, abs_ig, abs_split_event_class, abs_event_window, # Additional parameters from args
    mine_mvs, mine_mg, mine_rel, mine_sf, mine_e,      # Additional parameters from args
    general_results_summary_dir,                        # Directory for the overall summary CSV file
    model_type                                          # The model nickname
):
    """
    Executes the aggregation and evaluation process for a specific aggregation method and TIRP selection method.

    Parameters:
        aggregation_method (str): Method name for aggregation.
        num_tirps_for_selection: Number of TIRPs to select for aggregation.
        tirp_selection_method (str): TIRP selection method to use.
        tte_w_list (list): List of TTE window sizes to process serially.
        e_w_list (list): List of early warning values to process serially.
        tirp_models_base_dir (str): Directory containing all TIRP model prediction outputs (Stage 3 outputs).
        results_output_dir (str): Directory where the final evaluation result CSVs for the current run will be saved.
        dataset_name (str): Name of the dataset.
        fold_num (int): Fold number.
        abs_d_method (str): Abstraction discretization method.
        abs_b (int): Number of bins for abstraction.
        abs_ig (int): Interpolation gap for abstraction.
        abs_split_event_class (str): Whether to split data based on event class.
        abs_event_window (str): Window size for event/class split.
        mine_mvs (float): Mining minimum vertical support.
        mine_mg (int): Mining max gap.
        mine_rel (int): Mining number of relations.
        mine_sf (bool): Mining skip followers.
        mine_e (int): Mining epsilon.
        general_results_summary_dir (str): Directory where the summary CSV of all experiments will be saved.
        model_type (str): the model type (FCPM/CPML).
    """
    print(f"--- Starting Stage 5: Aggregation & Evaluation ---")
    print(f"Method: {aggregation_method}, TIRP Selection: {tirp_selection_method}")
    print(f"TTE Windows: {tte_w_list}, Early Warning: {e_w_list}")
    print(f"Input (Prediction Dirs): {tirp_models_base_dir}")
    print(f"Output (ROC CSVs Dir for this run): {results_output_dir}")
    print(f"Summary CSV Dir: {general_results_summary_dir}")
    start_total_time = time.time()

    # --- Step 1: Aggregate Predictions (Once per TIRP selection method) ---
    print(f"Step 1: Aggregating predictions using method '{aggregation_method}' with TIRP selection '{tirp_selection_method}'...")
    
    start_agg_time = time.time()
    base_dir = os.path.dirname(tirp_models_base_dir)
    tirp_scoring_file_path = os.path.join(base_dir, 'tirp_selection_scores.csv')

    # Ensure the directory for the summary CSV exists
    os.makedirs(general_results_summary_dir, exist_ok=True)

    if tirp_selection_method == 'all':
        # If 'all', no need to specify number of TIRPs, just use the method name
        tirp_selection_method_name = f'Binary_{tirp_selection_method}'
    else:
        # For other methods, include the number of TIRPs to select
        tirp_selection_method_name = f'Binary_{tirp_selection_method}#{num_tirps_for_selection}'
    print(f"Processing TIRP selection method: {tirp_selection_method_name}")

    aggregation = Aggregation(
        method_name=aggregation_method,
        prediction_output_dir=tirp_models_base_dir,
        tirp_selection_method = tirp_selection_method_name,
        scores_file_path=tirp_scoring_file_path
    )
    # aggregate_predictions() should return the base name or relevant identifier for the aggregated file
    base_file_name_for_eval = aggregation.aggregate_predictions() 
    end_agg_time = time.time()
    print(f"Aggregation for {tirp_selection_method_name} finished. Time: {end_agg_time - start_agg_time:.2f} seconds.")
# --- Step 2: Evaluate Aggregated Results (Loop over TTE_W and e_w combinations) ---
    print(f"Step 2: Evaluating aggregated predictions (ROC) for all TTE_W and e_w combinations in memory...")
    
    os.makedirs(results_output_dir, exist_ok=True) 
    
    start_eval_time = time.time()

    # Create the prefix exactly as the original filename format requires
    base_result_name = f'{model_type}_ts_{tirp_selection_method}_agg_{aggregation_method}'

    # Initialize Evaluate once per execution
    evaluation = Evaluate(
        prediction_output_dir=tirp_models_base_dir, 
        agg_method=aggregation_method,
        agg_file_name=base_file_name_for_eval
    )
    
    # Run the optimized evaluation (this generates ALL csv files, plots, and thresholds)
    results_dict = evaluation.evaluate_roc(
        tte_w_list=tte_w_list,
        e_w_list=e_w_list,
        output_dir=results_output_dir,
        base_result_name=base_result_name
    )

    end_eval_time = time.time()
    print(f"  Evaluation finished for all parameters. Time: {end_eval_time - start_eval_time:.2f} seconds.")

    # --- Save experiment parameters to summary CSV ---
    summary_csv_file_path = os.path.join(general_results_summary_dir, f"fold_num_{fold_num}_abs_{abs_d_method}_{abs_b}_{abs_ig}_KL_{mine_mvs}_{mine_mg}_{mine_rel}_{mine_sf}_{mine_e}_{model_type}_agg_{aggregation_method}_{tirp_selection_method}#{num_tirps_for_selection}.csv")
    
    file_exists = os.path.isfile(summary_csv_file_path)
    
    with open(summary_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        for TTE_window_size in tte_w_list:
            for early_warning_value in e_w_list:
                # Retrieve the specific AUC and AUPRC for this combination
                AUC, AUPRC = results_dict.get((TTE_window_size, early_warning_value), (0.0, 0.0))

                experiment_data_row = {
                    'dataset_name': dataset_name,
                    'fold_num': fold_num,
                    'abs_method': abs_d_method,
                    'bins': abs_b,
                    'abs_ig': abs_ig,
                    # 'split_event_class': abs_split_event_class,
                    # 'event_window': abs_event_window,
                    'KL_mvs': mine_mvs,
                    'KL_mg': mine_mg,
                    'KL_num_rel': mine_rel,
                    'KL_sf': mine_sf,
                    'KL_e': mine_e,
                    'model_type': model_type,
                    'agg_method': aggregation_method,
                    'tirp_selection_method': tirp_selection_method,
                    'num_tirps_for_selection': num_tirps_for_selection,
                    'TTE_WS': TTE_window_size,
                    'early_warning': early_warning_value,
                    'AUC': AUC,
                    'AUPRC': AUPRC,
                }
                
                fieldnames = list(experiment_data_row.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True # Ensure header is written only once
                
                writer.writerow(experiment_data_row)
                print(f"  Experiment parameters for TTE_W={TTE_window_size}, e_w={early_warning_value} appended to {summary_csv_file_path}")

    end_total_time = time.time()
    print(f"--- Finished Stage 5: Aggregation & Evaluation. Total Time: {end_total_time - start_total_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Run aggregation and evaluation for TIRP model predictions (Simplified).")

    # General Info
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--fold_num", required=True, type=int, help="Fold number for cross-validation.")

    # Abstraction Params
    parser.add_argument("--abs_d_method", required=True, help="Abstraction discretization method (e.g., SAX, KMEANS).")
    parser.add_argument("--abs_b", required=True, type=int, help="Number of bins for abstraction.")
    parser.add_argument("--abs_ig", required=True, type=int, help="Interpolation gap for abstraction.")
    parser.add_argument("--abs_split_event_class", type=str, default="False", help="Whether to split data based on event class.")
    parser.add_argument("--abs_event_window", type=str, default="None", help="Window size for event/class split.")

    # Mining Params
    parser.add_argument("--mine_mvs", required=True, type=float, help="Minimum vertical support for mining TIRPs.")
    parser.add_argument("--mine_mg", required=True, type=int, help="Maximum gap allowed between symbols in a TIRP.")
    parser.add_argument("--mine_rel", required=True, type=int, help="Number of Allen's relations to consider.")
    parser.add_argument("--mine_sf", required=True, type=lambda x: (str(x).lower() == 'true'), help="Skip followers during mining (True/False).")
    parser.add_argument("--mine_e", required=True, type=int, help="Epsilon for temporal constraint satisfaction.")

    # Aggregation/Eval Params
    parser.add_argument("--agg_aggregation_method", required=True, help="Method for aggregating predictions (e.g., AVG, MAX).")
    parser.add_argument("--agg_num_tirps_for_selection", required=True, type=int, help="Number of TIRPs to select for aggregation.")
    
    # TIRP Selection Method
    parser.add_argument("--tirp_selection_method", required=True, help="TIRP selection method to use.")
    
    # TTE and Early Warning Lists (comma-separated)
    parser.add_argument("--TTE_W_list", required=True, help="Comma-separated list of TTE window sizes.")
    parser.add_argument("--e_w_list", required=True, help="Comma-separated list of early warning values.")
    parser.add_argument("--model_type", required=True, help="Nickname of the model (FCPM/CPML).")

    # Paths
    parser.add_argument("--prediction_base_dir", required=True, help="Base directory containing Stage 3 (TIRP model) prediction outputs.")
    parser.add_argument("--output_csv_path", required=True, help="Directory to save the output ROC CSVs for this specific run.")
    parser.add_argument("--results_dir", required=True, help="Directory to save the general experiment summary CSV.")

    args = parser.parse_args()

    # Parse the comma-separated lists
    tte_w_list = [int(x.strip()) for x in args.TTE_W_list.split(',') if x.strip()]
    e_w_list = [int(x.strip()) for x in args.e_w_list.split(',') if x.strip()]

    run_single_aggregation_evaluation(
        aggregation_method=args.agg_aggregation_method,
        num_tirps_for_selection=args.agg_num_tirps_for_selection,
        tirp_selection_method=args.tirp_selection_method,
        tte_w_list=tte_w_list,
        e_w_list=e_w_list,
        tirp_models_base_dir=args.prediction_base_dir,
        results_output_dir=args.output_csv_path, # Directory for specific ROC CSVs
        dataset_name=args.dataset_name,
        fold_num=args.fold_num,
        abs_d_method=args.abs_d_method,
        abs_b=args.abs_b,
        abs_ig=args.abs_ig,
        abs_split_event_class=args.abs_split_event_class,
        abs_event_window=args.abs_event_window,
        mine_mvs=args.mine_mvs,
        mine_mg=args.mine_mg,
        mine_rel=args.mine_rel,
        mine_sf=args.mine_sf,
        mine_e=args.mine_e,
        general_results_summary_dir=args.results_dir, # Directory for the overall summary CSV
        model_type=args.model_type
    )

    sys.exit(0)