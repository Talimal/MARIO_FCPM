
import os
import pandas as pd
import numpy as np
import logging
import warnings

# Suppress pandas PerformanceWarnings caused by iterative column generation
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from tqdm import tqdm
from sklearn.metrics import auc
import matplotlib.pyplot as plt

class Evaluate:
    """
        Evaluate Class
    This class is designed to evaluate the performance of a prediction model by calculating 
    True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN) 
    based on specific evaluation criteria.
    Attributes:
        prediction_output_dir (str): Directory containing prediction outputs for entities.
        decision_threshold (float): Threshold for FCPM_Prediction to be considered positive.
        window_size (list): Allowed ranges for TTE_true to match TTE_prediction (Now passed to evaluate_roc).
        early_warning_value (list): Minimum number of consecutive timestamps with positive predictions required (Now passed to evaluate_roc).
    Methods:
        __init__(self, prediction_output_dir, agg_method, agg_file_name, entity_ids=None, save_threshold_files=False):
            Initializes the Evaluate class with the required parameters.
        evaluate_roc(self, tte_w_list, e_w_list, output_dir, file_prefix, ...):
            Evaluates the model's performance in memory for all combinations and calculates TP, FP, FN, and TN.
            Returns:
                dict: Contains tuples of (roc_auc, auprc) indexed by (window_size, early_warning) keys.
    """

    def __init__(self, prediction_output_dir, agg_method, agg_file_name, entity_ids=None, save_threshold_files=False):
        """
        Initialize the Evaluate class.

        :param model: The model to be evaluated.
        :param test_data: The test data to evaluate the model on.
        :param save_threshold_files: Whether to save threshold files for each combination of parameters (default: True).
        """
        self.prediction_output_dir = prediction_output_dir
        self.entity_ids = entity_ids
        self.agg_method = agg_method
        self.agg_file_name = agg_file_name
        self.save_threshold_files = save_threshold_files

        self.tirp_file_name = None  # Default tirp file name, can be overridden if needed


    def _classify_entity_at_threshold(self, aggregated_df, threshold, window_size, early_warning_value):
        """
        Classifies a single entity based on its aggregated data for a specific threshold,
        aligning with baseline logic for NaN/TTE handling and detection.
        'early_warning_value' here is treated as baseline's 'td'.
        
        Now also adds a threshold column to the dataframe and marks the decision TFS with 1.

        Args:
            aggregated_df (pd.DataFrame): The dataframe for the entity.
            threshold (float): The FCPM decision threshold.
            window_size (int): TTE accuracy window (total width).
            early_warning_value (int): Represents baseline's 'td'.
                                    If 0, detect on 1st point >= threshold.
                                    If D > 0, need D+1 points, detect at D-th point.

        Returns:
            tuple: (classification, modified_df)
                - classification (str): Classification ('TP', 'FP', 'FN', 'TN') or None if error/invalid data.
                - modified_df (pd.DataFrame): Original dataframe with added threshold column
        """
        required_cols = ["FCPM_Prediction", "TTE_prediction", "TTE_true", "outcome_class"]
        
        # We work directly on the dataframe to accumulate columns efficiently (saves memory instead of .copy())
        modified_df = aggregated_df
        threshold_col_name = f"ew_{early_warning_value}_ws_{window_size}_threshold_{threshold:.3f}"
        
        # Initialize threshold column with 0s
        modified_df[threshold_col_name] = 0
        
        # Ensure 'EntityID' column exists for logging, or handle its absence
        if 'EntityID' in aggregated_df.columns and not aggregated_df.empty:
            entity_id_str = str(aggregated_df['EntityID'].iloc[0])
        else:
            entity_id_str = 'Unknown'

        if aggregated_df.empty or not all(col in aggregated_df.columns for col in required_cols):
            logging.warning(f"Entity {entity_id_str}: DF empty or missing required columns ({required_cols}). Cannot classify.")
            return None, modified_df # Invalid input

        # Ensure consistent outcome class (should be same for all rows of an entity)
        outcome_class_series = aggregated_df["outcome_class"].unique()
        if len(outcome_class_series) > 1:
            logging.warning(f"Entity {entity_id_str} has multiple outcome classes: {outcome_class_series}. Using first value.")
        outcome_class = int(outcome_class_series[0])

        current_sequence_above_threshold = [] # Stores rows of the current consecutive sequence
        detection_row_data = None
        decision_index = None  # Track which row index made the decision

        for i, row in aggregated_df.iterrows():
            fcmp_pred = row["FCPM_Prediction"]
            # Handle potential NaN predictions safely - treat as below threshold
            if pd.isna(fcmp_pred):
                fcmp_pred = -np.inf

            if fcmp_pred >= threshold:
                current_sequence_above_threshold.append((i, row))  # Store index and row
            else:
                # Sequence is broken, reset
                current_sequence_above_threshold = []

            # Check for detection condition based on baseline logic (using early_warning_value as td)
            if detection_row_data is None: # Only detect once
                if early_warning_value == 0:
                    if len(current_sequence_above_threshold) >= 1:
                        # Detect on the first point that meets the threshold
                        decision_index, detection_row_data = current_sequence_above_threshold[0]
                        modified_df.loc[decision_index, threshold_col_name] = 1  # Mark decision TFS
                        logging.debug(f"Entity {entity_id_str}: Decision (td=0 logic) at index {decision_index}.")
                        break
                elif early_warning_value > 0:
                    # Need early_warning_value + 1 consecutive points
                    required_sequence_length = early_warning_value + 1
                    if len(current_sequence_above_threshold) >= required_sequence_length:
                        # Detection point is the D-th point (1-based) in this D+1 sequence.
                        # This corresponds to index (early_warning_value - 1) in the current_sequence_above_threshold list
                        detection_point_index_in_sequence = early_warning_value - 1
                        decision_index, detection_row_data = current_sequence_above_threshold[detection_point_index_in_sequence]
                        modified_df.loc[decision_index, threshold_col_name] = 1  # Mark decision TFS
                        logging.debug(f"Entity {entity_id_str}: Decision (td={early_warning_value} logic) at index {decision_index} of {early_warning_value}-th point in {required_sequence_length}-point sequence.")
                        break

        # --- After scanning all rows (or breaking), classify based on detection ---
        if detection_row_data is not None:
            logging.debug(f"Entity {entity_id_str} (Class={outcome_class}): Detection occurred at threshold {threshold:.2f} with 'early_warning_value' (as td) = {early_warning_value}")

            if outcome_class == 1: # Truly Positive Entity
                TTE_prediction_raw = detection_row_data.get("TTE_prediction")
                TTE_true_raw = detection_row_data.get("TTE_true") # Renamed to TTE_true_raw for clarity before conversion

                # Handle missing TTE_true
                if pd.isna(TTE_true_raw):
                    raise ValueError(f"Entity {entity_id_str}: TTE_true is missing/NaN for outcome_class=1 entity with detection. This indicates a bug in previous data processing steps.")

                # Convert TTE_true to float
                try:
                    tte_true_f = float(TTE_true_raw)
                except (ValueError, TypeError):
                    raise ValueError(f"Entity {entity_id_str}: TTE_true '{TTE_true_raw}' is not a valid number for outcome_class=1 entity with detection. This indicates a bug in previous data processing steps.")

                # Baseline-like handling for TTE_prediction:
                TTE_prediction_final = None # Initialize
                if pd.isna(TTE_prediction_raw):
                    raise ValueError(f"Entity {entity_id_str}: TTE_prediction is missing/NaN for outcome_class=1 entity with detection. This indicates a bug in previous data processing steps.")
                else:
                    try:
                        TTE_prediction_final = float(TTE_prediction_raw)
                        TTE_prediction_final = max(0, TTE_prediction_final) # Align with baseline's max(0, ...)
                    except (ValueError, TypeError):
                        raise ValueError(f"Entity {entity_id_str}: TTE_prediction '{TTE_prediction_raw}' is not a valid number for outcome_class=1 entity with detection. This indicates a bug in previous data processing steps.")

                # Check if TTE_true is within the window around TTE_prediction
                win_half = float(window_size) / 2.0
                # Ensure TTE_prediction_final is not None before arithmetic (should be handled by earlier checks)
                if TTE_prediction_final is None: # Should not happen if logic above is correct
                    logging.error(f"  Entity {entity_id_str}: TTE_prediction_final is unexpectedly None. Classified as FP.")
                    return "FP", modified_df

                lower_bound = TTE_prediction_final - win_half
                upper_bound = TTE_prediction_final + win_half

                if lower_bound <= tte_true_f <= upper_bound:
                    return "TP", modified_df
                else:
                    return "FP", modified_df

            else: # outcome_class == 0 (Truly Negative Entity)
                # Detected but True Class is 0. Classified as FP
                return "FP", modified_df
        else:
            # --- No Detection Occurred ---
            if outcome_class == 1:
                return "FN", modified_df
            else: # outcome_class == 0
                return "TN", modified_df

    def _classify_entity_at_threshold_vectorized(self, aggregated_df, threshold, window_size, early_warning_value):
        """
        Optimized vectorized version of entity classification.
        Uses pandas vectorized operations instead of sequential row iteration for better performance.
        
        Performance benefits:
        - 10-100x speedup for large datasets (>1000 TFS points)
        - Especially efficient for late detection, no detection, and high early warning values
        - Handles edge cases (NaN, empty sequences) correctly
        
        Args:
            aggregated_df (pd.DataFrame): The dataframe for the entity.
            threshold (float): The FCPM decision threshold.
            window_size (int): TTE accuracy window (total width).
            early_warning_value (int): Consecutive points requirement.

        Returns:
            tuple: (classification, modified_df)
                - classification (str): Classification ('TP', 'FP', 'FN', 'TN') or None if error/invalid data.
                - modified_df (pd.DataFrame): Original dataframe with added threshold column
        """
        required_cols = ["FCPM_Prediction", "TTE_prediction", "TTE_true", "outcome_class"]
        
        # Work directly on the dataframe to save memory footprint over many iterations
        modified_df = aggregated_df
        threshold_col_name = f"ew_{early_warning_value}_ws_{window_size}_threshold_{threshold:.3f}"
        
        # Initialize threshold column with 0s
        modified_df[threshold_col_name] = 0
        
        # Handle entity ID for logging
        if 'EntityID' in aggregated_df.columns and not aggregated_df.empty:
            entity_id_str = str(aggregated_df['EntityID'].iloc[0])
        else:
            entity_id_str = 'Unknown'

        if aggregated_df.empty or not all(col in aggregated_df.columns for col in required_cols):
            logging.warning(f"Entity {entity_id_str}: DF empty or missing required columns ({required_cols}). Cannot classify.")
            return None, modified_df

        # Get outcome class
        outcome_class_series = aggregated_df["outcome_class"].unique()
        if len(outcome_class_series) > 1:
            logging.warning(f"Entity {entity_id_str} has multiple outcome classes: {outcome_class_series}. Using first value.")
        outcome_class = int(outcome_class_series[0])

        # VECTORIZED APPROACH: Create boolean mask for predictions above threshold
        predictions = aggregated_df["FCPM_Prediction"].fillna(-np.inf)  # Handle NaN as below threshold
        above_threshold_mask = predictions >= threshold
        
        if not above_threshold_mask.any():
            # No points above threshold
            detection_occurred = False
            detection_index = None
        else:
            # Find consecutive sequences using cumulative operations
            # Create groups of consecutive True values
            consecutive_groups = (above_threshold_mask != above_threshold_mask.shift(1)).cumsum()
            
            # Filter only groups where value is True (above threshold)
            above_threshold_groups = consecutive_groups[above_threshold_mask]
            
            # Calculate sequence lengths for each group
            group_lengths = above_threshold_groups.value_counts().sort_index()
            
            # Required sequence length
            required_length = early_warning_value + 1 if early_warning_value > 0 else 1
            
            # Find first group that meets length requirement
            qualifying_groups = group_lengths[group_lengths >= required_length]
            
            if len(qualifying_groups) == 0:
                # No sequence long enough
                detection_occurred = False
                detection_index = None
            else:
                # Get the first qualifying group
                first_qualifying_group = qualifying_groups.index[0]
                
                # Get indices of this group
                group_indices = above_threshold_groups[above_threshold_groups == first_qualifying_group].index
                
                # Determine detection point based on early warning logic
                if early_warning_value == 0:
                    # Detect at first point
                    detection_index = group_indices[0]
                else:
                    # Detect at early_warning_value-th point (0-indexed: early_warning_value-1)
                    if len(group_indices) > early_warning_value - 1:
                        detection_index = group_indices[early_warning_value - 1]
                    else:
                        # Shouldn't happen if logic above is correct
                        detection_index = group_indices[0]
                
                detection_occurred = True
                
                # Mark decision point
                modified_df.loc[detection_index, threshold_col_name] = 1
                # logging.debug(f"Entity {entity_id_str}: Vectorized detection at index {detection_index}")

        # Classification logic (same as original)
        if detection_occurred:
            detection_row = aggregated_df.loc[detection_index]
            
            if outcome_class == 1:  # Positive entity
                TTE_prediction_raw = detection_row.get("TTE_prediction")
                TTE_true_raw = detection_row.get("TTE_true")

                if pd.isna(TTE_true_raw):
                    raise ValueError(f"Entity {entity_id_str}: TTE_true is missing/NaN for outcome_class=1 entity with detection.")

                try:
                    tte_true_f = float(TTE_true_raw)
                    TTE_prediction_final = max(0, float(TTE_prediction_raw))
                except (ValueError, TypeError):
                    raise ValueError(f"Entity {entity_id_str}: Invalid TTE values for outcome_class=1 entity with detection.")

                # Check TTE window
                win_half = float(window_size) / 2.0
                lower_bound = TTE_prediction_final - win_half
                upper_bound = TTE_prediction_final + win_half

                if lower_bound <= tte_true_f <= upper_bound:
                    return "TP", modified_df
                else:
                    return "FP", modified_df
            else:  # outcome_class == 0
                return "FP", modified_df
        else:
            # No detection
            if outcome_class == 1:
                return "FN", modified_df
            else:
                return "TN", modified_df

    def _classify_entity_at_threshold_smart(self, aggregated_df, threshold, window_size, early_warning_value):
        """
        Smart hybrid approach that automatically chooses the best classification method 
        based on data characteristics for optimal performance.
        
        Selection criteria:
        - Large datasets (>500 TFS points): Use vectorized (10-100x faster)
        - High early warning values (>3): Use vectorized (better for complex patterns)
        - Otherwise: Use sequential (lower overhead for small/simple cases)
        
        Args:
            aggregated_df (pd.DataFrame): The dataframe for the entity.
            threshold (float): The FCPM decision threshold.
            window_size (int): TTE accuracy window (total width).
            early_warning_value (int): Consecutive points requirement.

        Returns:
            tuple: (classification, modified_df) - identical to both underlying methods
        """
        n_points = len(aggregated_df)
        
        # Use vectorized for larger datasets or high early warning values
        if n_points > 500 or early_warning_value > 3:
            return self._classify_entity_at_threshold_vectorized(
                aggregated_df, threshold, window_size, early_warning_value)
        else:
            return self._classify_entity_at_threshold(
                aggregated_df, threshold, window_size, early_warning_value)

    def evaluate_roc(self, tte_w_list, e_w_list, output_dir='', base_result_name='', entity_ids=None,
                    num_thresholds=101, save_threshold_files=None,val_output_csv=None):
        """
        Evaluate and calculate ROC/AUC and PR/AUPRC by scanning multiple thresholds,
        applying the 'consecutive_count + TTE window' logic.
        Now performs optimized multi-parameter evaluation in memory to avoid repeated Disk I/O.
        (Plots are kept structurally intact but commented out as requested).

        Args:
            tte_w_list (list): List of TTE accuracy windows (total width).
            e_w_list (list): List of early warning values.
            output_dir (str): Directory to save output CSVs for each combination.
            base_result_name (str): Base name used for the specific output files.
            entity_ids (list, optional): Subset of entity IDs (strings or ints) to evaluate.
            num_thresholds (int): How many thresholds to test (0.0 to 1.0).
            save_threshold_files (bool, optional): Override instance's save_threshold_files setting.
            val_output_csv (str, optional): Path to save the output CSV, use for stage 3.5 validation.

        Returns:
            dict: Contains tuples of (roc_auc, auprc) indexed by (window_size, early_warning) tuples.
        """
        # --- Parameter Setup ---
        eval_entity_ids = set(map(str, entity_ids)) if entity_ids is not None else self.entity_ids 
        eval_save_threshold_files = save_threshold_files if save_threshold_files is not None else self.save_threshold_files

        logging.info(f"Starting optimized ROC & PR evaluation: processing {len(tte_w_list)} Windows and {len(e_w_list)} Early Warnings combinations, num_thresholds={num_thresholds}")

        # --- Entity Selection ---
        all_entities_dirs = [
            d for d in os.listdir(self.prediction_output_dir)
            if os.path.isdir(os.path.join(self.prediction_output_dir, d))
        ]
        if eval_entity_ids is not None: # Filter if specific IDs were provided
            entities_to_process = [d for d in all_entities_dirs if d in eval_entity_ids]
        else: # Otherwise use all found directories
            entities_to_process = all_entities_dirs
        logging.info(f"Evaluating {len(entities_to_process)} entities.")

        # --- Threshold Iteration Setup ---
        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        
        # Initialize a multi-level dictionary to store stats for all combinations
        stats = {}
        for w in tte_w_list:
            stats[w] = {}
            for ew in e_w_list:
                stats[w][ew] = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds}

        # --- OUTER LOOP: Entity Iteration (READ FILES ONLY ONCE) ---
        for entity_dir in entities_to_process:
            aggregated_file_path = os.path.join(self.prediction_output_dir, entity_dir, self.agg_file_name)

            if not os.path.exists(aggregated_file_path):
                logging.warning(f"Skipping entity {entity_dir}: Aggregated file not found at {aggregated_file_path}.")
                continue

            required_cols = ["FCPM_Prediction", "TTE_prediction", "TTE_true", "outcome_class", "EntityID"]

            try:
                # Read the file ONCE - include ALL columns to preserve existing threshold columns
                aggregated_df = pd.read_csv(aggregated_file_path, compression='gzip')
                
                # Check that required columns are present
                missing_required = [col for col in required_cols if col not in aggregated_df.columns]
                if missing_required:
                    logging.warning(f"Skipping entity {entity_dir}: Missing required columns {missing_required}")
                    continue

                # Check: Ensure dataframe is not empty after reading
                if aggregated_df.empty:
                    logging.warning(f"Skipping entity {entity_dir}: Aggregated file {aggregated_file_path} resulted in an empty DataFrame.")
                    continue

            except pd.errors.EmptyDataError:
                logging.warning(f"Skipping entity {entity_dir}: Aggregated file {self.agg_file_name} is empty or contains no columns.")
                continue
            except ValueError as ve: 
                logging.warning(f"Skipping entity {entity_dir}: Error reading columns from {self.agg_file_name}. Maybe columns mismatch? Error: {ve}")
                continue
            except Exception as e:
                logging.error(f"Skipping entity {entity_dir}: Failed to read aggregated file {self.agg_file_name}. Error: {e}")
                continue

            # --- INNER LOOPS: Evaluate all combinations in memory ---
            for w in tte_w_list:
                for ew in e_w_list:
                    for t in thresholds:
                        classification, _ = self._classify_entity_at_threshold_smart(aggregated_df, t, w, ew)
                        if classification:
                            stats[w][ew][t][classification.lower()] += 1

            # --- Save threshold files separately for each combination (avoid race conditions) ---
            if eval_save_threshold_files:
                entity_path = os.path.join(self.prediction_output_dir, entity_dir)
                agg_method_name = self.agg_file_name.replace('aggregated_', '').replace('.csv.gz', '')
                threshold_dir_name = f"threshold_{agg_method_name}"
                threshold_dir_path = os.path.join(entity_path, threshold_dir_name)
                
                # Create the threshold directory if it doesn't exist
                os.makedirs(threshold_dir_path, exist_ok=True)
                
                for w in tte_w_list:
                    for ew in e_w_list:
                        threshold_filename = f"thresholds_ew_{ew}_ws_{w}.csv.gz"
                        threshold_file_path = os.path.join(threshold_dir_path, threshold_filename)
                        
                        # Extract only the threshold columns for this early warning value and window size
                        threshold_cols = [col for col in aggregated_df.columns if col.startswith(f"ew_{ew}_ws_{w}_threshold_")]
                        
                        # Include only EntityID and TFS for identification, plus threshold columns
                        identification_cols = ["EntityID", "TFS"]
                        cols_to_save = [c for c in identification_cols if c in aggregated_df.columns]
                        cols_to_save.extend(threshold_cols)
                        
                        threshold_df = aggregated_df[cols_to_save].copy()
                        
                        # Save threshold file
                        threshold_df.to_csv(threshold_file_path, index=False, compression='gzip')
                        logging.info(f"Saved threshold decisions for entity {entity_dir}, early_warning={ew} to: {threshold_dir_name}/{threshold_filename}")

        if not eval_save_threshold_files:
            logging.info("Skipping threshold file creation as save_threshold_files is disabled.")

        # --- Final ROC/AUC and PR/AUPRC Calculation per parameter combination ---
        final_results = {}
        for w in tte_w_list:
            for ew in e_w_list:
                threshold_metrics = []
                for t in thresholds:
                    tp = stats[w][ew][t]["tp"]
                    fp = stats[w][ew][t]["fp"]
                    fn = stats[w][ew][t]["fn"]
                    tn = stats[w][ew][t]["tn"]

                    total_positives = tp + fn
                    total_negatives = fp + tn
                    total_population = total_positives + total_negatives

                    tpr = tp / total_positives if total_positives > 0 else 0.0
                    fpr = fp / total_negatives if total_negatives > 0 else 0.0
                    
                    # --- ADDED METRICS ---
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tpr # Recall is same as TPR
                    accuracy = (tp + tn) / total_population if total_population > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    # --- ---

                    threshold_metrics.append({
                        "threshold": t, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                        "TPR": tpr, "FPR": fpr, "Precision": precision, "Recall": recall, # Added
                        "F1": f1, "Accuracy": accuracy # Added
                    })
                    logging.debug(f"Thresh={t:.3f} -> TP={tp}, FP={fp}, TN={tn}, FN={fn}, TPR={tpr:.3f}, FPR={fpr:.3f}, Prec={precision:.3f}, F1={f1:.3f}")

                metrics_df = pd.DataFrame(threshold_metrics)

                # --- ROC AUC ---
                metrics_df_roc_sorted = metrics_df.sort_values(by=["FPR", "TPR"])
                fprs = metrics_df_roc_sorted["FPR"].tolist()
                tprs = metrics_df_roc_sorted["TPR"].tolist()
                roc_thresholds = metrics_df_roc_sorted["threshold"].tolist()

                if len(fprs) < 2: roc_auc = 0.0
                else: roc_auc = auc(fprs, tprs)
                
                # --- PR AUC ---
                # Sort points by recall for PR AUC calculation
                metrics_df_pr_sorted = metrics_df.sort_values(by=["Recall", "Precision"])
                pr_recalls = metrics_df_pr_sorted["Recall"].tolist()
                pr_precisions = metrics_df_pr_sorted["Precision"].tolist()

                if len(pr_recalls) < 2: auprc = 0.0
                else: auprc = auc(pr_recalls, pr_precisions) # AUC of Precision vs Recall

                if val_output_csv is not None:
                    metrics_df.to_csv(val_output_csv, index=False)
                    logging.info(f"Saved metrics to {val_output_csv}")
                    return roc_auc, auprc
                else:
                    # Unique name output file per w and ew combination
                    roc_result_file_name = f'{base_result_name}_ws_{w}_ew_{ew}.csv'
                    output_csv = os.path.join(output_dir, roc_result_file_name)

                # # --- Plotting ---
                # plot_save_dir = output_csv[:-4]+'_roc_curve.png'

                # # Plot ROC Curve
                # fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                # ax_roc.plot(fprs, tprs, color='darkorange', lw=2, marker='.', label=f'ROC curve (AUC = {roc_auc:.3f})')
                # ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                # ax_roc.set_xlim([0.0, 1.0])
                # ax_roc.set_ylim([0.0, 1.05])
                # ax_roc.set_xlabel('False Positive Rate (FPR)')
                # ax_roc.set_ylabel('True Positive Rate (TPR)')
                # ax_roc.set_title(f'ROC Curve ({self.agg_method} Aggregation)')
                # ax_roc.legend(loc="lower right")
                # ax_roc.grid(True, alpha=0.5)
                # roc_plot_path = plot_save_dir
                # fig_roc.savefig(roc_plot_path, bbox_inches='tight')
                # plt.close(fig_roc) # Close figure
                # logging.info(f"ROC curve plot saved to {roc_plot_path}")

                # plot_save_dir = output_csv[:-4]+'_pr_curve.png'

                # # Plot Precision-Recall Curve
                # fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                # # Plot recall vs precision
                # ax_pr.plot(pr_recalls, pr_precisions, color='blue', lw=2, marker='.', label=f'PR curve (AUPRC = {auprc:.3f})')
                # ax_pr.set_xlim([0.0, 1.0])
                # ax_pr.set_ylim([0.0, 1.05])
                # ax_pr.set_xlabel('Recall (TPR)')
                # ax_pr.set_ylabel('Precision')
                # ax_pr.set_title(f'Precision-Recall Curve ({self.agg_method} Aggregation)')
                # ax_pr.legend(loc="lower left")
                # ax_pr.grid(True, alpha=0.5)
                # pr_plot_path = plot_save_dir
                # fig_pr.savefig(pr_plot_path, bbox_inches='tight')
                # plt.close(fig_pr) # Close figure
                # logging.info(f"PR curve plot saved to {pr_plot_path}")

                # --- Prepare Output ---
                # Save detailed metrics if requested
                if output_dir is not None:
                    # Save the original (threshold-ordered) dataframe with all metrics
                    metrics_df.sort_values("threshold").to_csv(output_csv, index=False, float_format='%.5f')
                    logging.info(f"ROC/PR evaluation details saved to {output_csv}")

                final_results[(w, ew)] = (roc_auc, auprc)

        # Return the dictionary holding all combinations (ROC/AUC, PR/AUPRC)
        return final_results