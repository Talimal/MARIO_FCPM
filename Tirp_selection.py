import pandas as pd
import os
from SKL.Tirp_detection import TIRPDetector
from SKL.Tirp_new import TIRP , TirpMatrix
import time
import numpy as np
import random 
from config import *


def get_tirp_prefixes(full_tirp):
    """
    Decomposes a full TIRP into a list of its STI-based prefixes.
    Each prefix is returned as a new TIRP object, containing the
    corresponding symbols and a TirpMatrix with the relevant relations list.
    (Simplified - assumes valid input TIRP object with expected interface).

    Args:
        full_tirp (TIRP): The original, complete TIRP object. Assumed to have
                          _symbols, size, and _tirp_matrix attributes.
                          _tirp_matrix assumed to have get_relations() method
                          returning a flat list of relations.

    Returns:
        list[TIRP]: A list of TIRP objects representing prefixes (length 1 to full).
                    Returns an empty list if input is invalid/empty or size < 1.
    """
    prefix_list = []
    # Assume attributes exist and are valid (minimal check based on previous code)
    if not hasattr(full_tirp, '_tirp_matrix') or not hasattr(full_tirp._tirp_matrix, 'get_relations'):
         return prefix_list

    original_symbols = full_tirp._symbols
    # Get the flat list of all relations from the original matrix object
    all_relations_flat = full_tirp._tirp_matrix.get_relations()

    # Generate prefixes from length 1 up to full_tirp.size
    for k in range(1, full_tirp.size + 1):
        prefix_symbols = original_symbols[:k]

        # Calculate the number of relations for a prefix of size k
        # This corresponds to the length of the slice needed from the flat list
        num_prefix_relations = (k * (k - 1)) // 2

        # Get the subset of relations from the flat list
        # Handle k=1 case where num_prefix_relations is 0
        prefix_relations_flat = all_relations_flat[:num_prefix_relations] if k > 1 else []

        # --- Construct the new TIRP object for the prefix ---
        prefix_tirp = TIRP() # Create a new empty TIRP object
        prefix_tirp._symbols = prefix_symbols
        prefix_tirp.size = k # Explicitly set the size

        # Create and populate the TirpMatrix for the prefix
        prefix_matrix = TirpMatrix()
        # Directly set the internal relations list - use a copy to avoid aliasing issues
        prefix_matrix._relations = list(prefix_relations_flat) # Create a copy of the slice
        prefix_matrix._size = k - 1 # Set the matrix size (number of symbols - 1)

        # Assign the new matrix object to the prefix TIRP
        prefix_tirp._tirp_matrix = prefix_matrix

        # Optional: Copy other relevant attributes if needed
        # if hasattr(full_tirp, 'some_other_attribute'):
        #    prefix_tirp.some_other_attribute = full_tirp.some_other_attribute

        prefix_list.append(prefix_tirp)
        # --- End construction ---

    return prefix_list

def get_total_entities(data_path):
    """ Reads number of entities from KL file header. """
    # No extra checks added, assumes file exists and is valid
    with open(data_path, 'r') as file:
        lines = file.readlines()
    # Assumes header is present and correct on line 2 (index 1)
    number_of_entities = lines[1].split(",")[-1]
    return int(number_of_entities)



class TirpSelection:
    """
    Filters TIRPs, calculates scores, prefix metrics (normalized VS, HS, MMD),
    adds selection flags based on ranking per method, and returns results.
    (Simplified - assumes valid inputs and interfaces).
    """
    # Added max_tirps_limit back to __init__
    def __init__(self, event_symbol, scoring_methods=['vertical_support']):
        """
        Initializes the TirpSelection instance.

        Args:
            event_symbol (str or int): Symbol for filtering TIRPs.
            scoring_methods (list[str], optional): Scoring methods to calculate. Defaults to ['vertical_support'].
                Available methods:
                - 'vertical_support': Score based on vertical support
                - 'diff_vertical_support': Score based on difference in vertical support
                - 'mean_squared_mmd': Score based on mean squared MMD prefix differences
                - 'mean_squared_vs': Score based on mean squared VS prefix differences
                - 'random': Random selection (hierarchical)
                - 'all': Union of all other selection methods (creates Binary_all column)
        """
        self.event_symbol = str(event_symbol)
        self.scoring_methods = scoring_methods
        self.scoring_methods_to_calculate = [m for m in scoring_methods if m not in ['random', 'all']]
        self.select_random = 'random' in scoring_methods


    # --- Private methods for calculating specific scores ---
    def _score_vertical_support(self, tirp):
        """ Calculates VS score for the full TIRP. """
        return tirp.get_vertical_support()

    def _score_diff_vertical_support(self, tirp, prefix_stats_cls0, prefix_stats_cls1):
        """ Calculates the absolute difference in normalized VS for the last prefix. """
        last_prefix_repr = None
        if tirp.size > 1:
            all_prefixes_objs = get_tirp_prefixes(tirp); last_prefix_obj = None
            for p in reversed(all_prefixes_objs):
                if p.size == tirp.size - 1: last_prefix_obj = p; break
            if last_prefix_obj: last_prefix_repr = last_prefix_obj.to_string()
        if last_prefix_repr:
            default_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0}
            stats0 = prefix_stats_cls0.get(last_prefix_repr, default_stats)
            stats1 = prefix_stats_cls1.get(last_prefix_repr, default_stats)
            vs_norm_cls0 = stats0.get('VS', 0.0); vs_norm_cls1 = stats1.get('VS', 0.0)
            diff = vs_norm_cls1 - vs_norm_cls0 # Use absolute difference
            return round(diff, 3)
        else: return 0.0


    def _score_diff_horizontal_support(self, tirp, prefix_stats_cls0, prefix_stats_cls1):
        """ Calculates the absolute difference in normalized HS for the last prefix. """
        last_prefix_repr = None
        if tirp.size > 1:
            all_prefixes_objs = get_tirp_prefixes(tirp); last_prefix_obj = None
            for p in reversed(all_prefixes_objs):
                if p.size == tirp.size - 1: last_prefix_obj = p; break
            if last_prefix_obj: last_prefix_repr = last_prefix_obj.to_string()
        if last_prefix_repr:
            default_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0}
            stats0 = prefix_stats_cls0.get(last_prefix_repr, default_stats)
            stats1 = prefix_stats_cls1.get(last_prefix_repr, default_stats)
            hs_norm_cls0 = stats0.get('HS', 0.0); hs_norm_cls1 = stats1.get('HS', 0.0)
            diff = hs_norm_cls1 - hs_norm_cls0 # Use absolute difference
            return round(diff, 3)
        else: return 0.0
    
    def _score_diff_mean_duration(self, tirp, prefix_stats_cls0, prefix_stats_cls1):
        """ Calculates the absolute difference in MMD for the last prefix. """
        last_prefix_repr = None
        if tirp.size > 1:
            all_prefixes_objs = get_tirp_prefixes(tirp); last_prefix_obj = None
            for p in reversed(all_prefixes_objs):
                if p.size == tirp.size - 1: last_prefix_obj = p; break
            if last_prefix_obj: last_prefix_repr = last_prefix_obj.to_string()
        if last_prefix_repr:
            default_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0}
            stats0 = prefix_stats_cls0.get(last_prefix_repr, default_stats)
            stats1 = prefix_stats_cls1.get(last_prefix_repr, default_stats)
            mmd_cls0 = stats0.get('MMD', 0.0); mmd_cls1 = stats1.get('MMD', 0.0)
            diff = mmd_cls1 - mmd_cls0
            return round(diff, 3)
        else: return 0.0

    def _score_mean_squared_mmd_prefix_diff(self, tirp, prefix_stats_cls0, prefix_stats_cls1):
        """
        Calculates the Mean of Squared Differences (MSD) of prefix Mean Mean Durations (MMD)
        between class 1 and class 0 for all prefixes of the given TIRP.
        """
        sum_of_squared_differences = 0.0
        num_prefixes_processed = 0

        # Assume get_tirp_prefixes function is available and works as before
        # It returns a list of TIRP objects, each representing a prefix
        tirp_prefixes = get_tirp_prefixes(tirp)

        if not tirp_prefixes:
            return 0.0 # No prefixes, so mean squared difference is 0

        default_prefix_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0}

        for prefix_obj in tirp_prefixes[:-1]:
            # Assume prefix_obj has a to_string() method for representation
            prefix_repr = prefix_obj.to_string()

            stats_cls0 = prefix_stats_cls0.get(prefix_repr, default_prefix_stats)
            stats_cls1 = prefix_stats_cls1.get(prefix_repr, default_prefix_stats)

            mmd_cls0 = stats_cls0.get('MMD', 0.0) # Use .get for safety, default to 0.0
            mmd_cls1 = stats_cls1.get('MMD', 0.0) # Use .get for safety, default to 0.0
            
            difference = mmd_cls1 - mmd_cls0
            sum_of_squared_differences += (difference ** 2)
            num_prefixes_processed += 1

        if num_prefixes_processed == 0: # Should not happen if tirp_prefixes is not empty
            return 0.0
            
        mean_squared_difference = sum_of_squared_differences / num_prefixes_processed
        
        return round(mean_squared_difference, 3)
    
    def _score_mean_squared_vs_prefix_diff(self, tirp, prefix_stats_cls0, prefix_stats_cls1):
        """
        Calculates the Mean of Squared Differences (MSD) of prefix normalized VS
        between class 1 and class 0 for all prefixes of the given TIRP.
        """
        sum_of_squared_differences = 0.0
        num_prefixes_processed = 0
        tirp_prefixes = get_tirp_prefixes(tirp) # Assume this function is available
        if not tirp_prefixes:
            return 0.0

        default_prefix_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0} # 'VS' here is VS_norm

        for prefix_obj in tirp_prefixes[:-1]: # Exclude the last prefix
            prefix_repr = prefix_obj.to_string() 

            stats_cls0 = prefix_stats_cls0.get(prefix_repr, default_prefix_stats)
            stats_cls1 = prefix_stats_cls1.get(prefix_repr, default_prefix_stats)

            # Recall that 'VS' in prefix_stats_clsX already stores the normalized VS
            vs_cls0 = stats_cls0.get('VS', 0.0)
            vs_cls1 = stats_cls1.get('VS', 0.0)

            difference = vs_cls1 - vs_cls0 # Class 1 - Class 0
            sum_of_squared_differences += difference
            num_prefixes_processed += 1

        if num_prefixes_processed == 0:
            return 0.0

        mean_squared_difference = sum_of_squared_differences / num_prefixes_processed
        return round(mean_squared_difference, 3)

    # Add other private scoring methods here


    def calculate_scores(self, tirps_list, class0_data_path=None, class1_data_path=None, detection_params=None):
            """
            Calculates scores and prefix metrics, adds selection flags, returns DataFrame.
            """
            if not tirps_list: return pd.DataFrame() # Handle empty input list early

            # Step 1: Filter TIRPs
            filtered_tirps = []
            for tirp in tirps_list:
                last_symbol = str(tirp._symbols[-1]) if tirp._symbols else None; tirp_size = tirp.size
                if not (tirp_size > 2 and last_symbol == self.event_symbol): continue
                filtered_tirps.append(tirp)

            if not filtered_tirps: return pd.DataFrame() # Return empty if nothing passed filter

            # Step 1b: Prepare for hierarchical random selection using MAX_TIRPS_FOR_SELECTION
            hierarchical_random_selections = {}
            if self.select_random:
                # Use a dedicated, SEED-initialized RNG so the random TIRP selection
                # is reproducible across runs without disturbing the global random state.
                rng = random.Random(SEED)
                selection_limits = sorted(MAX_TIRPS_FOR_SELECTION, reverse=True)  # Sort in descending order
                n_filtered = len(filtered_tirps)

                if selection_limits and n_filtered > 0:
                    # Start with the largest limit
                    largest_limit = selection_limits[0]
                    n_to_sample = min(largest_limit, n_filtered)

                    if n_to_sample > 0:
                        # Select the largest set first
                        current_pool = rng.sample(filtered_tirps, n_to_sample)
                        hierarchical_random_selections[largest_limit] = set(tirp.to_string() for tirp in current_pool)

                        # For each smaller limit, select from the previous pool
                        for limit in selection_limits[1:]:
                            if limit < len(current_pool):
                                current_pool = rng.sample(current_pool, limit)
                            hierarchical_random_selections[limit] = set(tirp.to_string() for tirp in current_pool)

            # Step 1c: Collect Unique Prefixes from *all* filtered TIRPs
            unique_prefixes_map = {}
            for tirp in filtered_tirps:
                prefixes = get_tirp_prefixes(tirp)
                for prefix in prefixes:
                    prefix_repr = prefix.to_string()
                    if prefix_repr not in unique_prefixes_map: unique_prefixes_map[prefix_repr] = prefix
            unique_prefix_list = list(unique_prefixes_map.values())


            # Step 2: Run Detection & Extract Prefix Stats
            prefix_stats_cls0 = {}; prefix_stats_cls1 = {}
            if class0_data_path and class1_data_path and detection_params:
                def get_prefix_stats_from_detection(data_path, prefixes_to_detect, params, total_entities_in_class):
                    prefix_stats = {}
                    if not prefixes_to_detect or not data_path: return prefix_stats
                    detector = TIRPDetector(time_intervals_path=data_path, num_relations=params.get('relations'), max_gap=params.get('max_gap'), epsilon=params.get('epsilon'), output_path=None, print_instances=False, one_size_tirp=True)
                    detected_prefixes_list = detector.run_detection(prefixes_to_detect, use_parallel=False)
                    for detected_prefix in detected_prefixes_list:
                        prefix_repr = detected_prefix.to_string()
                        vs_raw = detected_prefix.get_vertical_support()
                        hs_raw = detected_prefix.calculate_mean_horizontal_support()
                        mmd_raw = detected_prefix.calculate_mean_mean_duration()
                        vs_normalized = round(vs_raw / total_entities_in_class, 3) if total_entities_in_class > 0 else 0.0
                        prefix_stats[prefix_repr] = {'VS': vs_normalized, 'HS': round(hs_raw, 3), 'MMD': round(mmd_raw, 3)}
                    return prefix_stats
                total_entities_cls0 = get_total_entities(class0_data_path); total_entities_cls1 = get_total_entities(class1_data_path)
                prefix_stats_cls0 = get_prefix_stats_from_detection(class0_data_path, unique_prefix_list, detection_params, total_entities_cls0)
                prefix_stats_cls1 = get_prefix_stats_from_detection(class1_data_path, unique_prefix_list, detection_params, total_entities_cls1)
            else:
                print("Warning: Data paths or detection params missing, cannot calculate prefix metrics.")


            # Step 3: Compile Final Results Data
            results_data = []
            for tirp in filtered_tirps:
                tirp_size = tirp.size; vertical_support = round(self._score_vertical_support(tirp), 3); mean_horizontal_support = round(tirp.calculate_mean_horizontal_support(), 3); mean_mean_duration = round(tirp.calculate_mean_mean_duration(), 3); tirp_representation = tirp.to_string()

                # Calculate actual scores for the full TIRP
                tirp_scores = {}
                for method_name in self.scoring_methods_to_calculate:
                    score_key = f'Score_{method_name}'
                    score_value = None
                    
                    if method_name == 'vertical_support':
                        score_value = vertical_support
                    elif method_name == 'diff_vertical_support':
                        score_value = self._score_diff_vertical_support(tirp, prefix_stats_cls0, prefix_stats_cls1)
                    elif method_name == 'diff_horizontal_support':
                        score_value = self._score_diff_horizontal_support(tirp, prefix_stats_cls0, prefix_stats_cls1)
                    elif method_name == 'diff_mean_duration':
                        score_value = self._score_diff_mean_duration(tirp, prefix_stats_cls0, prefix_stats_cls1)
                    elif method_name == 'mean_squared_mmd':
                        score_value = self._score_mean_squared_mmd_prefix_diff(tirp, prefix_stats_cls0, prefix_stats_cls1)
                    elif method_name == 'mean_squared_vs': 
                        score_value = self._score_mean_squared_vs_prefix_diff(tirp, prefix_stats_cls0, prefix_stats_cls1)
                    
                    tirp_scores[score_key] = score_value

                # Gather prefix metrics
                current_tirp_prefix_metrics_cls0 = {}; current_tirp_prefix_metrics_cls1 = {}
                prefixes = get_tirp_prefixes(tirp)
                for prefix in prefixes:
                    prefix_repr = prefix.to_string(); default_stats = {'VS': 0.0, 'HS': 0.0, 'MMD': 0.0}
                    stats0 = prefix_stats_cls0.get(prefix_repr, default_stats); stats1 = prefix_stats_cls1.get(prefix_repr, default_stats)
                    current_tirp_prefix_metrics_cls0[prefix_repr] = stats0; current_tirp_prefix_metrics_cls1[prefix_repr] = stats1

                result_entry = {'TIRP_Object': tirp, 'TIRP_Representation': tirp_representation, 'Size': tirp_size, 'Vertical_Support': vertical_support, 'Mean_Horizontal_Support': mean_horizontal_support, 'Mean_Mean_Duration': mean_mean_duration, 'Prefix_Metrics_Cls0': current_tirp_prefix_metrics_cls0, 'Prefix_Metrics_Cls1': current_tirp_prefix_metrics_cls1, **tirp_scores}
                results_data.append(result_entry)


            # Step 4: Create DataFrame
            if not results_data: return pd.DataFrame()
            scores_df = pd.DataFrame(results_data)

            # Step 5: Add Selection Flags
            # Add flags for hierarchical random selection
            if self.select_random:
                for limit in MAX_TIRPS_FOR_SELECTION:
                    column_name = f'Binary_random#{limit}'
                    selected_tirps = hierarchical_random_selections.get(limit, set())
                    scores_df[column_name] = scores_df['TIRP_Representation'].apply(lambda x: 1 if x in selected_tirps else 0)

            # Add flags for score-based selection using MAX_TIRPS_FOR_SELECTION limits
            for method_name in self.scoring_methods_to_calculate:
                score_col = f'Score_{method_name}'
                
                if score_col in scores_df.columns:
                    # Sort TIRPs by score for this method
                    sorted_df = scores_df.sort_values(by=score_col, ascending=False, na_position='last')
                    
                    # Create selection columns for each limit in MAX_TIRPS_FOR_SELECTION
                    total_tirps = len(sorted_df)
                    for limit in MAX_TIRPS_FOR_SELECTION:
                        selected_col = f'Binary_{method_name}#{limit}'
                        scores_df[selected_col] = 0  # Initialize all to 0
                        
                        if limit > 0:
                            # Take minimum of limit and total available TIRPs
                            actual_limit = min(limit, total_tirps)
                            top_n_indices = sorted_df.head(actual_limit).index
                            scores_df.loc[top_n_indices, selected_col] = 1

            # Add "all" method: Binary_all column (union of all other selection methods)
            if 'all' in self.scoring_methods:
                # Find all Binary columns (excluding the Binary_all column we're about to create)
                binary_columns = [col for col in scores_df.columns if col.startswith('Binary_') and col != 'Binary_all']
                
                if binary_columns:
                    # Create Binary_all as the union (logical OR) of all other Binary columns
                    scores_df['Binary_all'] = scores_df[binary_columns].max(axis=1)
                else:
                    # If no other Binary columns exist, set all to 0
                    scores_df['Binary_all'] = 0

            return scores_df