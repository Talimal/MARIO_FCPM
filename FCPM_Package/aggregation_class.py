import pandas as pd
import numpy as np
import os
import warnings # Import warnings
from pandas.errors import EmptyDataError

class Aggregation:
    """
    Class for performing various aggregation methods on TIRP predictions for a single entity DataFrame.
    Optionally filters TIRPs based on a specified selection method defined in a scores file before aggregation.
    """

    def __init__(self, method_name, prediction_output_dir, tirp_selection_method=None, scores_file_path=None, top_percentage=0.1):
        """
        Initializes the Aggregation class.

        Args:
            method_name (str): The aggregation method to use (e.g., "avg", "max", "min",
                               "avg_top_percentage", "VS_weighted_avg", "HS_weighted_avg", "MMD_weighted_avg").
            prediction_output_dir (str): Directory containing entity subdirectories with prediction CSVs.
            tirp_selection_method (str, optional): The column name in the scores file that indicates TIRP
                                                   selection (e.g., 'vertical_support').
                                                   If None, no selection-based filtering is applied. Defaults to None.
            scores_file_path (str, optional): Path to the CSV file containing TIRP scores and the selection column.
                                              Required if tirp_selection_method is provided or weighted avg methods are used.
            top_percentage (float, optional): The top percentage of predictions (based on FCPM_Prediction)
                                              to consider for the "avg_top_percentage" method. E.g., 0.2 for top 20%.
        """
        self.supported_methods = [
            "max", "min", "avg", "avg_top_percentage",
            "VS_weighted_avg", "HS_weighted_avg", "MMD_weighted_avg", "weighted_avg"
        ]
        if method_name not in self.supported_methods:
            raise ValueError(f"Unsupported aggregation method '{method_name}'. "
                             f"Supported methods are: {self.supported_methods}")

        self.method_name = method_name
        self.prediction_output_dir = prediction_output_dir
        self.scores_file_path = scores_file_path
        self.top_percentage = top_percentage

        # Handle tirp_selection_method (singular)
        if tirp_selection_method is not None and not isinstance(tirp_selection_method, str):
             raise TypeError("tirp_selection_method must be a string or None.")
        self.tirp_selection_method = tirp_selection_method # Store the single string or None

        # --- Validation and Pre-loading ---
        # Scores file is required if using weighted average OR if a selection method is specified
        requires_scores_file = bool(self.tirp_selection_method) or \
                               self.method_name in ["VS_weighted_avg", "HS_weighted_avg", "MMD_weighted_avg", "weighted_avg"]

        if requires_scores_file:
            if self.scores_file_path is None or not os.path.exists(self.scores_file_path):
                raise FileNotFoundError(f"Scores file not found at {self.scores_file_path}. Required for weighted averages or TIRP selection filtering.")
            try:
                self.scores_df = pd.read_csv(self.scores_file_path)

                # Define required columns based on usage
                required_score_cols = ['TIRP_Representation']
                if self.method_name == "VS_weighted_avg": required_score_cols.append('Vertical_Support')
                if self.method_name == "HS_weighted_avg": required_score_cols.append('Mean_Horizontal_Support')
                if self.method_name == "MMD_weighted_avg": required_score_cols.append('Mean_Mean_Duration')
                if self.method_name == "weighted_avg":
                    # For weighted_avg, determine the score column based on TIRP selection method
                    if self.tirp_selection_method:
                        if self.tirp_selection_method.startswith('Binary_'):
                            # Extract the base method name (e.g., 'random', 'all', 'vertical_support')
                            base_method = self.tirp_selection_method.replace('Binary_', '')
                            if '#' in base_method:
                                base_method = base_method.split('#')[0]
                            # Map to corresponding Score column
                            if base_method in ['random', 'all']:
                                required_score_cols.append('Vertical_Support')
                            else:
                                score_col = f'Score_{base_method}'
                                required_score_cols.append(score_col)
                        else:
                            # If not Binary_ format, use Vertical_Support as default
                            required_score_cols.append('Vertical_Support')
                    else:
                        # No selection method specified, use Vertical_Support as default
                        required_score_cols.append('Vertical_Support')
                # Add the single selection method column if provided
                if self.tirp_selection_method: required_score_cols.append(self.tirp_selection_method)

                # Ensure uniqueness
                required_score_cols = list(set(required_score_cols))

                missing_score_cols = [col for col in required_score_cols if col not in self.scores_df.columns]
                if missing_score_cols:
                    raise ValueError(f"Missing required columns in scores file '{self.scores_file_path}': {missing_score_cols}")

                # Keep only necessary columns
                self.scores_df = self.scores_df[required_score_cols]

            except Exception as e:
                raise ValueError(f"Error loading or processing scores file '{self.scores_file_path}': {e}")
        else:
            self.scores_df = None # No scores needed

        if self.method_name == "avg_top_percentage":
            if top_percentage is None or not (0 < top_percentage <= 1):
                raise ValueError("Parameter 'top_percentage' (between 0 and 1 exclusive/inclusive) is required for method 'avg_top_percentage'.")
        # --- End Validation ---


    def _generate_output_basename(self, debug=False):
        """
        Build the common part of the aggregated-results filename.
        Set debug=True to print how the name was constructed.
        """
        parts = []
        if self.tirp_selection_method:
            parts.append(f'ts_{self.tirp_selection_method}')

        parts.append(self.method_name)

        if self.method_name == "avg_top_percentage":
            pct = int(self.top_percentage * 100)
            parts.append(f"{pct}pct")

        fname = f"aggregated_{'__'.join(parts)}.csv.gz"

        if debug:
            print("[DEBUG-basename] selection =", self.tirp_selection_method,
                  "| method =", self.method_name,
                  "| top_pct =", getattr(self, "top_percentage", None),
                  "→", fname)

        return fname


    def _create_and_save_dummy_file(self, entity_path, entity_id, outcome_class_val):
        """Helper function to create and save the dummy aggregated file."""
        base_output_filename = self._generate_output_basename()
        output_filename = os.path.join(entity_path, base_output_filename)

        # Use entity_id (already processed), outcome_class_val (captured or default)
        dummy_data = {
            "EntityID": [entity_id],
            "TFS": [0],
            "outcome_class": [outcome_class_val if outcome_class_val is not None else np.nan], # Use captured or NaN
            "FCPM_Prediction": [0.0],
            "TTE_prediction": [0.0], # Set TTE_prediction to 0 as requested
            "TTE_true": [0.0]
        }
        dummy_df = pd.DataFrame(dummy_data)
        final_cols_order = ["EntityID", "TFS", "outcome_class", 'FCPM_Prediction', 'TTE_prediction', 'TTE_true']
        dummy_df = dummy_df.reindex(columns=final_cols_order)

        try:
            dummy_df.to_csv(output_filename, index=False, compression='gzip')
            return True # Indicate success
        except Exception as e:
            print(f"    - Error saving dummy aggregated file {output_filename}: {e}")
            return False # Indicate failure

    def aggregate_predictions(self, entity_ids=None):
        """
        Aggregates predictions, applying optional filtering, and creates dummy files if needed.
        """
        all_items = os.listdir(self.prediction_output_dir)
        all_entities_dirs = [d for d in all_items if
                             os.path.isdir(os.path.join(self.prediction_output_dir, d))]

        if entity_ids is not None:
            entity_ids_str_set = {str(eid) for eid in entity_ids}
            all_entities_dirs = [d for d in all_entities_dirs if d in entity_ids_str_set]

        base_output_filename = None

        scores_data_for_merge = self.scores_df if hasattr(self, 'scores_df') and self.scores_df is not None else None
        requires_scores_file = bool(self.tirp_selection_method) or \
                               self.method_name in ["VS_weighted_avg", "HS_weighted_avg", "MMD_weighted_avg"]

        # Generate the base output filename once at the beginning
        base_output_filename = self._generate_output_basename()
        
        for entity_dir in all_entities_dirs:
            entity_path = os.path.join(self.prediction_output_dir, entity_dir)
            tirp_prediction_files = [
                f for f in os.listdir(entity_path)
                if f.endswith(".csv.gz") and not f.startswith("aggregated_")
            ]

            entity_dfs = []
            first_outcome_class = None # Variable to store the first outcome class found
            # print(f"\nProcessing Entity: {entity_dir}")

            # Attempt to determine EntityID type
            try: entity_id_processed = int(entity_dir)
            except ValueError: entity_id_processed = entity_dir

            for file in tirp_prediction_files:
                file_path = os.path.join(entity_path, file)
                tirp_name = file.replace(".csv.gz", "")
                try:
                    # Read the file, handle if it's completely empty
                    try:
                        df = pd.read_csv(file_path, compression='gzip')
                    except EmptyDataError:
                        # print(f"   - Skipping TIRP {tirp_name}: File is empty (EmptyDataError).")
                        continue # Skip if pandas says it's empty

                    # Try to capture the first outcome_class even from potentially skipped files
                    if first_outcome_class is None and not df.empty and 'outcome_class' in df.columns:
                        try:
                            # Ensure there's at least one non-NaN value before taking iloc[0]
                            valid_outcomes = df['outcome_class'].dropna()
                            if not valid_outcomes.empty:
                                first_outcome_class = valid_outcomes.iloc[0]
                        except IndexError:
                            pass # Ignore if column exists but is all NaN or empty after dropna

                    # Now apply skipping logic
                    if df.empty: # Double check emptiness just in case
                        # print(f"   - Skipping TIRP {tirp_name}: File is empty after read.")
                        continue
                    if len(df) == 1 and 'instance_ID' in df.columns and df['instance_ID'].iloc[0] == -1:
                        # print(f"  - Skipping TIRP {tirp_name}: File has single row with instance_ID == -1.")
                        continue # Skip aggregation for this file, but outcome class might have been captured

                    # If not skipped, add TIRP Representation and add to list
                    df['TIRP_Representation'] = tirp_name
                    entity_dfs.append(df)

                except Exception as e: # Catch other potential errors during read/processing
                    print(f"  - Error processing {file_path}: {e}")
                    continue

            # --- Handle Case: No valid data loaded for aggregation ---
            if not entity_dfs:
                print(f"  - No valid prediction CSV files found or kept for entity {entity_dir}. Creating dummy output.")
                self._create_and_save_dummy_file(entity_path, entity_id_processed, first_outcome_class)
                continue # Skip to the next entity

            # --- Process if valid data was found ---
            concatenated_df = pd.concat(entity_dfs, ignore_index=True)

            # Merge with scores data ONCE if needed
            # (Merge logic remains the same as previous version)
            if scores_data_for_merge is not None:
                if 'TIRP_Representation' in concatenated_df.columns:
                    original_rows = len(concatenated_df)
                    concatenated_df = pd.merge(concatenated_df, scores_data_for_merge, on='TIRP_Representation', how='left')
                    if original_rows != len(concatenated_df): print(f"  - WARNING: Merge changed row count for {entity_dir}.")
                    check_cols = []
                    wm = { "VS_weighted_avg": "Vertical_Support", "HS_weighted_avg": "Mean_Horizontal_Support", "MMD_weighted_avg": "Mean_Mean_Duration" }
                    if self.method_name == "weighted_avg":
                        # For weighted_avg, determine the weight column based on TIRP selection method
                        if self.tirp_selection_method:
                            if self.tirp_selection_method.startswith('Binary_'):
                                # Extract the base method name (e.g., 'random', 'all', 'vertical_support')
                                base_method = self.tirp_selection_method.replace('Binary_', '')
                                if '#' in base_method:
                                    base_method = base_method.split('#')[0]
                                # Map to corresponding Score column
                                if base_method in ['random', 'all']:
                                    check_cols.append('Vertical_Support')
                                else:
                                    score_col = f'Score_{base_method}'
                                    check_cols.append(score_col)
                            else:
                                # If not Binary_ format, use Vertical_Support as default
                                check_cols.append('Vertical_Support')
                        else:
                            # No selection method specified, use Vertical_Support as default
                            check_cols.append('Vertical_Support')
                    else:
                        wc = wm.get(self.method_name);
                        if wc: check_cols.append(wc)
                    if self.tirp_selection_method: check_cols.append(self.tirp_selection_method)
                    for col in list(set(check_cols)):
                        if col in concatenated_df.columns and concatenated_df[col].isnull().any():
                             mt = concatenated_df.loc[concatenated_df[col].isnull(), 'TIRP_Representation'].unique()
                             print(f"  - WARNING: Scores/Select missing for col '{col}', TIRPs: {list(mt)} in {entity_dir}. Filling 0.")
                             concatenated_df[col] = concatenated_df[col].fillna(0)
                        elif col not in concatenated_df.columns:
                             print(f"  - WARNING: Expected col '{col}' missing after merge for {entity_dir}. Creating/Filling 0.")
                             concatenated_df[col] = 0
                # else: # TIRP_Representation missing
                #      if requires_scores_file: continue


            # Apply TIRP selection filtering
            df_to_aggregate = concatenated_df
            if self.tirp_selection_method is not None:
                print(f"  - Applying TIRP selection filter: '{self.tirp_selection_method}'")
                if self.tirp_selection_method in df_to_aggregate.columns:
                    try:
                        flags = pd.to_numeric(df_to_aggregate[self.tirp_selection_method], errors='coerce').fillna(0)
                        df_to_aggregate = df_to_aggregate[flags == 1].copy()
                    except Exception as e:
                        print(f"   - Error applying filter {self.tirp_selection_method}: {e}. Skipping filter application.")
                        # Continue without filtering in case of error

                    if df_to_aggregate.empty:
                        print(f"    - No TIRPs selected by method '{self.tirp_selection_method}' for entity {entity_dir}. Creating dummy output.")
                        self._create_and_save_dummy_file(entity_path, entity_id_processed, first_outcome_class)
                        continue # Skip to next entity
                else:
                    print(f"    - WARNING: Selection column '{self.tirp_selection_method}' not found. Skipping filtering.")

            # Perform Aggregation
            print(f"  - Performing aggregation using method: '{self.method_name}'")
            aggregated_result = self.aggregate(df_to_aggregate)

            # Save results or create dummy if aggregation fails
            if aggregated_result is None or aggregated_result.empty:
                print(f"    - Aggregation resulted in empty DataFrame for entity {entity_dir}. Creating dummy output.")
                self._create_and_save_dummy_file(entity_path, entity_id_processed, first_outcome_class)
                continue # Skip to next entity

            # Save the actual aggregated result
            sorted_aggregated_result = aggregated_result.sort_values(by=["TFS"]).reset_index(drop=True)
            output_filename = os.path.join(entity_path, base_output_filename)
            try:
                sorted_aggregated_result.to_csv(output_filename, index=False, compression='gzip')
            except Exception as e:
                print(f"    - Error saving aggregated file {output_filename}: {e}")
        return base_output_filename






    def _weighted_average(self, series, weights):
        """Calculates weighted average, handling NaNs and zero sum of weights."""
        series_numeric = pd.to_numeric(series, errors='coerce')
        weights_numeric = pd.to_numeric(weights, errors='coerce').fillna(0)
        valid_indices = ~series_numeric.isnull()
        series_valid = series_numeric[valid_indices]
        weights_valid = weights_numeric[valid_indices]
        if weights_valid.sum() == 0 or len(series_valid) == 0:
            return np.nan
        try:
            return np.average(series_valid, weights=weights_valid)
        except Exception as e:
            print(f"    - Error during weighted average calculation: {e}")
            return np.nan

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate continuous-prediction CSVs that belong to a single entity.

        Parameters
        ----------
        data : pd.DataFrame
            Concatenated (and already selection-filtered) dataframe that must contain at least:
            - 'EntityID', 'TFS', 'outcome_class'
            - 'FCPM_Prediction', 'TTE_prediction'
            - 'TIRP_Representation'    (added earlier in the pipeline)

        Returns
        -------
        pd.DataFrame
            One row per (EntityID, TFS) with the aggregated prediction columns,
            the optional ground-truth `TTE_true`, **and** a semicolon-separated list
            of TIRPs that contributed to that row (`TIRPs_Used`).
        """
        # ---------- sanity checks ----------
        if data.empty:
            return pd.DataFrame()                      # nothing to aggregate

        required_cols = ["EntityID", "TFS", "outcome_class",
                        "FCPM_Prediction", "TTE_prediction"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            print(f"  - Missing columns: {missing}  ⇒  skipping aggregation")
            return pd.DataFrame()

        # guarantee presence of representation column
        if "TIRP_Representation" not in data.columns:
            data = data.copy()
            data["TIRP_Representation"] = "unknown"

        # cast predictions to numeric (coerce errors to NaN)
        data["FCPM_Prediction"] = pd.to_numeric(data["FCPM_Prediction"],
                                                errors="coerce")
        data["TTE_prediction"] = pd.to_numeric(data["TTE_prediction"],
                                            errors="coerce")

        grouping_cols = ["EntityID", "TFS", "outcome_class"]
        grouped = data.groupby(grouping_cols, observed=True)

        # ---------- choose aggregation strategy ----------
        if self.method_name == "avg":
            agg_df = grouped.agg(
                FCPM_Prediction=('FCPM_Prediction', 'mean'),
                TTE_prediction=('TTE_prediction', 'mean')
            ).reset_index()

        elif self.method_name in {"max", "min"}:
            idx_func = 'idxmax' if self.method_name == "max" else 'idxmin'
            # pick the row with max/min prediction inside each group
            idx = grouped["FCPM_Prediction"].apply(
                lambda s: getattr(s.dropna(), idx_func)() if not s.dropna().empty else np.nan
            ).dropna().astype(int)
            agg_df = data.loc[idx, grouping_cols + ["FCPM_Prediction",
                                                    "TTE_prediction"]].reset_index(drop=True)

        elif self.method_name == "avg_top_percentage":
            top_p = max(1e-4, self.top_percentage)     # avoid zero
            def _avg_top(group):
                group = group.dropna(subset=["FCPM_Prediction"])
                if group.empty:
                    return pd.Series([np.nan, np.nan], index=["FCPM_Prediction", "TTE_prediction"])
                n = max(1, int(np.ceil(len(group) * top_p)))
                top = group.nlargest(n, "FCPM_Prediction")
                return pd.Series([top["FCPM_Prediction"].mean(),
                                top["TTE_prediction"].mean()],
                                index=["FCPM_Prediction", "TTE_prediction"])
            agg_df = grouped.apply(_avg_top, include_groups=False).reset_index()

        elif self.method_name in {"VS_weighted_avg", "HS_weighted_avg", "MMD_weighted_avg", "weighted_avg"}:
            if self.method_name == "weighted_avg":
                # For weighted_avg, determine the weight column based on TIRP selection method
                if self.tirp_selection_method:
                    if self.tirp_selection_method.startswith('Binary_'):
                        # Extract the base method name (e.g., 'random', 'all', 'vertical_support')
                        base_method = self.tirp_selection_method.replace('Binary_', '')
                        if '#' in base_method:
                            base_method = base_method.split('#')[0]
                        # Map to corresponding Score column
                        if base_method in ['random', 'all']:
                            w_col = 'Vertical_Support'
                        else:
                            w_col = f'Score_{base_method}'
                    else:
                        # If not Binary_ format, use Vertical_Support as default
                        w_col = 'Vertical_Support'
                else:
                    # No selection method specified, use Vertical_Support as default
                    w_col = 'Vertical_Support'
            else:
                # For other weighted methods, use the predefined mapping
                weight_map = {"VS_weighted_avg": "Vertical_Support",
                            "HS_weighted_avg": "Mean_Horizontal_Support",
                            "MMD_weighted_avg": "Mean_Mean_Duration"}
                w_col = weight_map[self.method_name]
            
            if w_col not in data.columns:
                print(f"  - Weight column '{w_col}' missing – cannot aggregate.")
                return pd.DataFrame()

            def _wavg(group):
                w = group[w_col]
                if w.sum() <= 0:
                    # all weights zero or negative – degrade to arithmetic mean
                    return pd.Series([group["FCPM_Prediction"].mean(),
                                    group["TTE_prediction"].mean()],
                                    index=["FCPM_Prediction", "TTE_prediction"])
                return pd.Series([
                    np.average(group["FCPM_Prediction"], weights=w),
                    np.average(group["TTE_prediction"],   weights=w)
                ], index=["FCPM_Prediction", "TTE_prediction"])

            agg_df = grouped.apply(_wavg, include_groups=False).reset_index()

        else:
            raise ValueError(f"Unknown aggregation method '{self.method_name}'")

        # ---------- attach list of TIRPs used ----------
        # 1. Calculate the mean FCPM_Prediction per TIRP within each group
        tirp_grouping = grouping_cols + ["TIRP_Representation"]
        mean_tirp_probs = (
            data.groupby(tirp_grouping, observed=True)["FCPM_Prediction"]
                .mean()
                .reset_index()
        )

        # 2. Format the string for each unique TIRP (TIRP_Name (mean_probability))
        mean_tirp_probs["TIRP_With_Prob"] = (
            mean_tirp_probs["TIRP_Representation"].astype(str) + 
            " (" + 
            mean_tirp_probs["FCPM_Prediction"].round(4).astype(str) + 
            ")"
        )

        # 3. Concatenate all TIRPs belonging to the same grouping_cols
        tirps_used = (
            mean_tirp_probs.groupby(grouping_cols, observed=True)["TIRP_With_Prob"]
                .apply(lambda s: ";".join(sorted(s.dropna().unique())))
                .reset_index(name="TIRPs_Used")
        )
        result = pd.merge(agg_df, tirps_used,
                        on=grouping_cols, how="left")

        # ---------- attach ground-truth TTE if present ----------
        if "TTE_true" in data.columns:
            tte_true = (
                data[["EntityID", "TFS", "TTE_true"]]
                .drop_duplicates(subset=["EntityID", "TFS"])
            )
            # align dtypes before merging
            for col in ["EntityID", "TFS"]:
                result[col] = result[col].astype(tte_true[col].dtype)
            result = pd.merge(result, tte_true,
                            on=["EntityID", "TFS"], how="left")
        else:
            result["TTE_true"] = np.nan

        # ---------- final column order ----------
        ordered_cols = grouping_cols + [
            "FCPM_Prediction", "TTE_prediction", "TTE_true", "TIRPs_Used"
        ]
        missing_final = [c for c in ordered_cols if c not in result.columns]
        for c in missing_final:      # insure consistent schema
            result[c] = np.nan
        return result[ordered_cols].sort_values(["EntityID", "TFS"]).reset_index(drop=True)

